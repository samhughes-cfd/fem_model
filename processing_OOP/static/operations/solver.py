# processing_OOP/static/operations/solver.py

import time
import logging
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from typing import Optional, Dict, Callable
from pathlib import Path
from processing_OOP.solver_registry import LinearSolverRegistry

class SolveCondensedSystem:
    """High-performance solver for condensed FEM systems K_cond U_cond = F_cond."""
    
    _PRECONDITIONER_REGISTRY = {
        'amg': spla.LinearOperator,
        'ilu': spla.spilu,
        'jacobi': lambda A: spla.diags(1/A.diagonal())
    }
    
    def __init__(
        self,
        K_cond: sp.csr_matrix,
        F_cond: np.ndarray,
        solver_name: str,
        job_results_dir: str,
        preconditioner: Optional[str] = None,
        max_mem_gb: float = 10.0
    ):
        """
        Parameters
        ----------
        K_cond : sp.csr_matrix
            Condensed stiffness matrix in CSR format
        F_cond : np.ndarray
            Condensed force vector
        solver_name : str
            Name of solver from LinearSolverRegistry
        job_results_dir : str
            Directory for solver diagnostics
        preconditioner : Optional[str]
            Preconditioning strategy (None/'amg'/'ilu'/'jacobi')
        max_mem_gb : float
            Memory budget in gigabytes
        """
        self.K_cond = K_cond
        self.F_cond = F_cond
        self.solver_name = solver_name
        self.job_results_dir = Path(job_results_dir)
        self.preconditioner = preconditioner
        self.max_mem = max_mem_gb * 1e9
        self._validate_condensed_system()
        
        # Solver state tracking
        self.U_cond = None
        self.diagnostics = {
            'solve_phase': {},
            'residuals': [],
            'memory_usage': [],
            'condition_estimate': None,
            'solver_steps': []
        }
        
        self._configure_logging()
        self._solver_registry = self._get_solver_registry()

    def _configure_logging(self):
        """Initialize condensed system-specific logging."""
        self.job_results_dir.mkdir(exist_ok=True, parents=True)
        log_file = self.job_results_dir / "condensed_solver.log"

        self.logger = logging.getLogger(f"CondensedSolver.{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # Structured JSON logging
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        ))
        self.logger.addHandler(file_handler)

        # Console logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(
            "[%(levelname)s] %(message)s"
        ))
        self.logger.addHandler(console_handler)

    def _validate_condensed_system(self):
        """Validate condensed system integrity."""
        if not sp.isspmatrix_csr(self.K_cond):
            raise ValueError("Condensed matrix must be CSR format")
            
        if self.K_cond.shape[0] != self.F_cond.shape[0]:
            raise ValueError("Condensed system dimension mismatch")
            
        if self._estimate_memory() > self.max_mem:
            raise MemoryError("Condensed system exceeds memory budget")

    def _estimate_memory(self) -> float:
        """Predict memory requirements in bytes."""
        matrix_mem = (self.K_cond.data.nbytes + 
                     self.K_cond.indices.nbytes + 
                     self.K_cond.indptr.nbytes)
        vector_mem = self.F_cond.nbytes
        return matrix_mem + vector_mem + 1e8  # 100MB buffer

    def _get_solver_registry(self) -> Dict[str, Callable]:
        """Get solver registry with condensed system preconditioning."""
        registry = LinearSolverRegistry.get_solver_registry()
        
        if self.preconditioner:
            return self._apply_condensed_preconditioner(registry)
        return registry

    def _apply_condensed_preconditioner(self, registry: Dict) -> Dict:
        """Apply preconditioner to condensed system solvers."""
        M = self._PRECONDITIONER_REGISTRY[self.preconditioner](self.K_cond)
        
        def wrap_solver(solver_fn):
            def preconditioned_solver(A, b):
                return solver_fn(A, b, M=M)
            return preconditioned_solver
            
        return {name: wrap_solver(fn) for name, fn in registry.items()}

    def _estimate_condensed_condition(self) -> float:
        """Condition number estimation for condensed system."""
        try:
            # Sparse-friendly estimation
            norm_A = spla.norm(self.K_cond, ord=2)
            norm_invA = spla.norm(spla.inv(self.K_cond), ord=2)
            return norm_A * norm_invA
        except:
            self.logger.warning("Condition estimation failed")
            return np.inf

    def _solve_direct_condensed(self) -> np.ndarray:
        """Direct solver for condensed system."""
        self.logger.debug("Factorizing condensed system")
        start_factor = time.perf_counter()
        factors = spla.splu(self.K_cond)
        self.diagnostics['solve_phase']['factorization'] = time.perf_counter() - start_factor
        
        self.logger.debug("Solving condensed system")
        start_solve = time.perf_counter()
        U_cond = factors.solve(self.F_cond)
        self.diagnostics['solve_phase']['solution'] = time.perf_counter() - start_solve
        
        return U_cond

    def _solve_iterative_condensed(self, solver_fn: Callable) -> np.ndarray:
        """Iterative solver with condensed system tracking."""
        self.logger.debug("Starting iterative solve on condensed system")
        
        def residual_callback(xk: np.ndarray):
            residual = np.linalg.norm(self.K_cond @ xk - self.F_cond)
            self.diagnostics['residuals'].append(residual)
            
            if len(self.diagnostics['residuals']) % 10 == 0:
                self.logger.debug(
                    f"Iter {len(self.diagnostics['residuals'])} | "
                    f"Residual: {residual:.2e}"
                )

        start = time.perf_counter()
        U_cond, info = solver_fn(self.K_cond, self.F_cond, callback=residual_callback)
        
        if info != 0:
            raise RuntimeError(f"Condensed solver failed with code {info}")
            
        self.diagnostics['solve_phase']['total'] = time.perf_counter() - start
        return U_cond

    def _validate_condensed_solution(self, U_cond: np.ndarray) -> float:
        """Validate condensed solution residual."""
        residual = self.K_cond @ U_cond - self.F_cond
        residual_norm = np.linalg.norm(residual)
        self.logger.info(f"Condensed residual norm: {residual_norm:.2e}")
        return residual_norm

    def solve_condensed(self) -> Optional[np.ndarray]:
        """Execute full solve process for condensed system."""
        try:
            # Phase 1: Pre-solve analysis
            self.diagnostics['condition_estimate'] = self._estimate_condensed_condition()
            self.logger.info(
                f"Condensed condition estimate: {self.diagnostics['condition_estimate']:.2e}"
            )
            
            # Phase 2: Solver dispatch with registry validation
            if not LinearSolverRegistry.solver_exists(self.solver_name):
                available = LinearSolverRegistry.list_solvers()
                raise ValueError(f"Invalid solver '{self.solver_name}'. Available: {available}")
            
            if 'direct' in self.solver_name:
                self.U_cond = self._solve_direct_condensed()
            else:
                solver_fn = self._solver_registry[self.solver_name]
                self.U_cond = self._solve_iterative_condensed(solver_fn)
                
            # Phase 3: Post-solve validation
            residual = self._validate_condensed_solution(self.U_cond)
            if residual > 1e-6:
                self.logger.warning("High condensed residual - verify solution")
                
            return self.U_cond
            
        except Exception as exc:
            self.logger.error(f"Condensed solve failed: {str(exc)}")
            self._dump_condensed_failure_state()
            return None

    def _dump_condensed_failure_state(self):
        """Save condensed system state on failure."""
        sp.save_npz(self.job_results_dir / "failed_K_cond.npz", self.K_cond)
        np.save(self.job_results_dir / "failed_F_cond.npy", self.F_cond)
        self.logger.info("Condensed failure state saved")

    def generate_condensed_report(self) -> Dict:
        """Generate condensed system performance report."""
        return {
            'condensed_system': {
                'dofs': self.K_cond.shape[0],
                'nonzeros': self.K_cond.nnz,
                'density': self.K_cond.nnz / (self.K_cond.shape[0]**2),
                'sparsity_pattern': str(self.K_cond.get_shape())
            },
            'performance': {
                'time': self.diagnostics.get('solve_phase', {}),
                'condition': self.diagnostics.get('condition_estimate'),
                'residuals': self.diagnostics.get('residuals', []),
                'memory': self.diagnostics.get('memory_usage', [])
            },
            'validation': {
                'final_residual': self._validate_condensed_solution(self.U_cond) 
                if self.U_cond is not None else np.nan
            }
        }

    def plot_condensed_performance(self):
        """Generate condensed system-specific plots."""
        self._plot_condensed_residuals()
        self._plot_condensed_timings()

    def _plot_condensed_residuals(self):
        """Plot condensed system residual history."""
        if not self.diagnostics['residuals']:
            return
            
        plt.figure(figsize=(10, 6))
        plt.semilogy(self.diagnostics['residuals'], 'b-o', lw=2, markersize=4)
        plt.title(f"Condensed System Residuals ({self.solver_name})")
        plt.xlabel("Iteration")
        plt.ylabel("Residual Norm")
        plt.grid(True, which='both', linestyle='--')
        plt.savefig(self.job_results_dir / "condensed_residuals.png")
        plt.close()

    def _plot_condensed_timings(self):
        """Visualize condensed solve phase timings."""
        if not self.diagnostics['solve_phase']:
            return
            
        labels, values = zip(*self.diagnostics['solve_phase'].items())
        
        plt.figure(figsize=(10, 6))
        plt.barh(labels, values, color='teal')
        plt.title("Condensed Solve Phase Timings")
        plt.xlabel("Time (seconds)")
        plt.tight_layout()
        plt.savefig(self.job_results_dir / "condensed_timings.png")
        plt.close()
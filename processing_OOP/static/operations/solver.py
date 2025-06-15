# processing_OOP/static/operations/solver.py

import time
import logging
import numpy as np
import os
import scipy.sparse as sp
from typing import Optional, Dict, Callable
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator
import scipy.sparse.linalg as spla

from processing_OOP.solver_registry import LinearSolverRegistry, conjugate_gradient_solver

class SolveCondensedSystem:
    """
    High-performance solver for condensed FEM systems of the form:
        K_cond @ U_cond = F_cond
    """

    _PRECONDITIONER_REGISTRY: Dict[str, Callable] = {
        'amg': spla.LinearOperator,  # Placeholder
        'ilu': spla.spilu,
        'jacobi': lambda A: LinearOperator(
            A.shape,
            matvec=lambda x: np.where(
                np.abs(A.diagonal()) > 1e-14,
                (1.0 / A.diagonal()) * x,
                0.0
            ),
            dtype=A.dtype
        )
    }

    def __init__(
        self,
        K_cond: sp.csr_matrix,
        F_cond: np.ndarray,
        solver_name: str,
        job_results_dir: str,
        preconditioner: Optional[str] = None,
        max_mem_gb: float = 10.0
    ) -> None:
        self.K_cond: sp.csr_matrix = K_cond.astype(np.float64)
        self.F_cond: np.ndarray = F_cond.astype(np.float64)
        self.solver_name: str = solver_name
        self.job_results_dir: Path = Path(job_results_dir)
        self.preconditioner: Optional[str] = preconditioner
        self.max_mem: float = max_mem_gb * 1e9
        self._validate_condensed_system()

        self.U_cond: Optional[np.ndarray] = None
        self.diagnostics: Dict[str, any] = {
            'solve_phase': {},
            'residuals': [],
            'memory_usage': [],
            'condition_estimate': None,
            'solver_steps': []
        }

        self.logger: logging.Logger = self._init_logging()
        self._solver_registry: Dict[str, Callable] = self._get_solver_registry()

    def _init_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"SolveCondensedSystem.{id(self)}")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        log_path: Optional[str] = None
        if self.job_results_dir:
            logs_dir = self.job_results_dir.parent / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / "SolveCondensedSystem.log"

            try:
                file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
                file_handler.setFormatter(logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s (Module: %(module)s, Line: %(lineno)d)"
                ))
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"⚠️ Failed to create file handler for log: {e}")

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(stream_handler)

        if log_path:
            logger.debug(f"📁 Log file created at: {log_path}")

        return logger

    def _validate_condensed_system(self) -> None:
        if not sp.isspmatrix_csr(self.K_cond):
            raise ValueError("Condensed matrix must be CSR format")
        if self.K_cond.shape[0] != self.F_cond.shape[0]:
            raise ValueError("Condensed system dimension mismatch")
        if self._estimate_memory() > self.max_mem:
            raise MemoryError("Condensed system exceeds memory budget")

    def _estimate_memory(self) -> float:
        matrix_mem = (self.K_cond.data.nbytes + 
                      self.K_cond.indices.nbytes + 
                      self.K_cond.indptr.nbytes)
        vector_mem = self.F_cond.nbytes
        return matrix_mem + vector_mem + 1e8

    def _get_solver_registry(self) -> Dict[str, Callable]:
        registry = LinearSolverRegistry.get_solver_registry()
        return self._apply_condensed_preconditioner(registry) if self.preconditioner else registry

    def _apply_condensed_preconditioner(self, registry: Dict[str, Callable]) -> Dict[str, Callable]:
        M = self._PRECONDITIONER_REGISTRY[self.preconditioner](self.K_cond)
        def wrap_solver(solver_fn: Callable) -> Callable:
            def preconditioned_solver(A: sp.csr_matrix, b: np.ndarray, callback=None) -> np.ndarray:
                return solver_fn(A, b, M=M, callback=callback)
            return preconditioned_solver
        return {name: wrap_solver(fn) for name, fn in registry.items()}

    def _estimate_spectral_norm(self, A: sp.spmatrix, max_iter: int = 10) -> float:
        n = A.shape[0]
        x = np.random.rand(n)
        x /= np.linalg.norm(x)
        for _ in range(max_iter):
            x = A @ x
            norm = np.linalg.norm(x)
            x /= norm
        return norm

    def _estimate_condensed_condition(self) -> float:
        try:
            norm_A = self._estimate_spectral_norm(self.K_cond)
            n: int = self.K_cond.shape[0]
            e: np.ndarray = np.random.randn(n)
            e /= np.linalg.norm(e)
            x, info = spla.cg(self.K_cond, e, maxiter=30, tol=1e-4)
            if info != 0:
                self.logger.warning("CG solver failed during condition estimate")
                return np.inf
            cond_est = norm_A * np.linalg.norm(x)
            if not np.isfinite(cond_est) or cond_est <= 0:
                self.logger.warning("Non-physical condition number estimate")
                return np.inf
            return cond_est
        except Exception as e:
            self.logger.warning(f"Condition estimation failed: {e}")
            return np.inf

    def _solve_direct_condensed(self) -> np.ndarray:
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
        self.logger.debug("Starting iterative solve on condensed system")

        def residual_callback(xk: np.ndarray) -> None:
            residual = np.linalg.norm(self.K_cond @ xk - self.F_cond)
            self.diagnostics['residuals'].append(residual)
            if len(self.diagnostics['residuals']) % 10 == 0:
                self.logger.debug(f"Iter {len(self.diagnostics['residuals'])} | Residual: {residual:.2e}")

        start = time.perf_counter()
        U_cond, info = solver_fn(self.K_cond, self.F_cond, callback=residual_callback)
        if info != 0:
            raise RuntimeError(f"Condensed solver failed with code {info}")
        self.diagnostics['solve_phase']['total'] = time.perf_counter() - start
        return U_cond

    def _validate_condensed_solution(self, U_cond: np.ndarray) -> float:
        residual = self.K_cond @ U_cond - self.F_cond
        residual_norm = np.linalg.norm(residual)
        self.logger.info(f"Condensed residual norm: {residual_norm:.2e}")
        return residual_norm

    def solve_condensed(self) -> Optional[np.ndarray]:
        try:
            self.diagnostics['condition_estimate'] = self._estimate_condensed_condition()
            self.logger.info(f"Condensed condition estimate: {self.diagnostics['condition_estimate']:.2e}")

            if not LinearSolverRegistry.solver_exists(self.solver_name):
                raise ValueError(f"Invalid solver '{self.solver_name}'. Available: {LinearSolverRegistry.list_solvers()}")

            self.U_cond = self._solve_direct_condensed() if 'direct' in self.solver_name \
                else self._solve_iterative_condensed(self._solver_registry[self.solver_name])

            residual = self._validate_condensed_solution(self.U_cond)
            if residual > 1e-6:
                self.logger.warning("High condensed residual - verify solution")

            if self.U_cond.shape != (self.K_cond.shape[0],):
                raise ValueError("Shape mismatch in U_cond")
            if self.U_cond.dtype != np.float64:
                raise TypeError("U_cond must be float64")

            report = self.generate_condensed_report()
            self.logger.info("📊 Condensed Solver Report:")
            for section, contents in report.items():
                self.logger.info(f"--- {section.upper()} ---")
                for key, value in contents.items():
                    if isinstance(value, dict):
                        for subkey, subval in value.items():
                            self.logger.info(f"{key} - {subkey}: {subval:.4e}" if isinstance(subval, (float, np.float64)) else f"{key} - {subkey}: {subval}")
                    elif isinstance(value, list):
                        self.logger.info(f"{key}: {[f'{float(v):.4e}' for v in value]}")
                    else:
                        self.logger.info(f"{key}: {value:.4e}" if isinstance(value, (float, np.float64)) else f"{key}: {value}")

            return self.U_cond

        except Exception as exc:
            self.logger.error(f"Condensed solve failed: {str(exc)}")
            self._dump_condensed_failure_state()
            return None

    def _dump_condensed_failure_state(self) -> None:
        sp.save_npz(self.job_results_dir / "failed_K_cond.npz", self.K_cond)
        np.save(self.job_results_dir / "failed_F_cond.npy", self.F_cond)
        self.logger.info("Condensed failure state saved")

    def generate_condensed_report(self) -> Dict[str, Dict[str, float]]:
        return {
            'condensed_system': {
                'dofs': int(self.K_cond.shape[0]),
                'nonzeros': int(self.K_cond.nnz),
                'density': float(self.K_cond.nnz / (self.K_cond.shape[0]**2)),
                'sparsity_pattern': str(self.K_cond.get_shape())
            },
            'performance': {
                'time': self.diagnostics.get('solve_phase', {}),
                'condition': self.diagnostics.get('condition_estimate'),
                'residuals': self.diagnostics.get('residuals', []),
                'memory': self.diagnostics.get('memory_usage', [])
            }
        }

    def plot_condensed_performance(self) -> None:
        self._plot_condensed_residuals()
        self._plot_condensed_timings()

    def _plot_condensed_residuals(self) -> None:
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

    def _plot_condensed_timings(self) -> None:
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
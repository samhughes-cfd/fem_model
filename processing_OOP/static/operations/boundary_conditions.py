# processing_OOP\static\operations\boundary_conditions.py

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
import logging
import os
import time
from typing import Tuple, Optional

class ModifyGlobalSystem:
    """High-performance boundary condition applier with numerical stabilization and diagnostics.
    
    Features:
    - Penalty method implementation with auto-scaling
    - Numerical stabilization for ill-conditioned systems
    - Precision-controlled matrix operations (float64)
    - Detailed sparse matrix diagnostics
    """
    
    def __init__(
        self,
        K_global: csr_matrix,
        F_global: np.ndarray,
        job_results_dir: Optional[str] = None,
        fixed_dofs: Optional[np.ndarray] = None
    ):
        """
        Parameters
        ----------
        K_global : csr_matrix
            Global stiffness matrix in CSR format
        F_global : np.ndarray
            Global force vector
        job_results_dir : Optional[str]
            Directory for diagnostic logs
        fixed_dofs : Optional[np.ndarray]
            Array of DOF indices to constrain (default: first 6 DOFs)
        """
        self.K_orig = K_global.astype(np.float64)
        self.F_orig = F_global.astype(np.float64)
        self.job_results_dir = job_results_dir
        self.fixed_dofs = fixed_dofs if fixed_dofs is not None else np.arange(6)
        self.K_mod = None
        self.F_mod = None
        self.penalty = None
        self._init_logging()
        self._validate_inputs()

    def _init_logging(self):
        """Configure hierarchical logging system."""
        self.logger = logging.getLogger(f"BoundaryConditionApplier.{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # File handler for detailed diagnostics
        if self.job_results_dir:
            os.makedirs(self.job_results_dir, exist_ok=True)
            log_path = os.path.join(self.job_results_dir, "boundary_conditions.log")
            file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            file_format = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s "
                "(Module: %(module)s, Line: %(lineno)d)"
            )
            file_handler.setFormatter(file_format)
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)

        # Console handler for critical messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

    def _validate_inputs(self):
        """Ensure input matrices meet requirements."""
        if not isinstance(self.K_orig, csr_matrix):
            raise TypeError("K_global must be a scipy.sparse.csr_matrix")
            
        if self.F_orig.ndim != 1:
            raise ValueError("F_global must be a 1D array")
            
        if self.K_orig.shape[0] != self.F_orig.shape[0]:
            raise ValueError("Matrix/vector dimension mismatch")

    def apply_boundary_conditions(self) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Execute full boundary condition application process."""
        start_time = time.perf_counter()
        self.logger.info("🔧 Applying stabilized boundary conditions...")
        
        try:
            self._convert_matrices()
            self._compute_penalty()
            self._apply_penalty_method()
            self._apply_stabilization()
            self._finalize_system()
            self._log_diagnostics()
        except Exception as e:
            self.logger.critical(f"❌ Boundary condition application failed: {str(e)}", exc_info=True)
            raise RuntimeError("Boundary condition application failed") from e
        
        exec_time = time.perf_counter() - start_time
        self.logger.info(f"✅ Boundary conditions applied in {exec_time:.2f}s")
        return self.K_mod, self.F_mod, self.fixed_dofs

    def _convert_matrices(self):
        """Convert matrices to working formats with precision control."""
        self.K_work = self.K_orig.astype(np.float64).tolil()
        self.F_work = self.F_orig.copy().astype(np.float64)

    def _compute_penalty(self):
        """Calculate dynamic penalty value."""
        diag = self.K_work.diagonal()
        self.penalty = np.float64(1e36) * (diag.max() if diag.any() else 1e36)
        self.logger.debug(f"Computed penalty value: {self.penalty:.2e}")

    def _apply_penalty_method(self):
        """Apply penalty method to fixed DOFs."""
        # Zero out rows/columns
        self.K_work[self.fixed_dofs, :] = 0.0
        self.K_work[:, self.fixed_dofs] = 0.0
        
        # Set diagonal entries
        for dof in self.fixed_dofs:
            self.K_work[dof, dof] = self.penalty
            
        # Zero force vector entries
        self.F_work[self.fixed_dofs] = 0.0

    def _apply_stabilization(self):
        """Add numerical stabilization to matrix."""
        stabilization = diags(
            [np.float64(1e-10) * self.penalty],
            [0],
            shape=self.K_work.shape,
            format='lil',
            dtype=np.float64
        )
        self.K_work = self.K_work + stabilization

    def _finalize_system(self):
        """Convert to final matrix formats."""
        self.K_mod = self.K_work.tocsr().astype(np.float64)
        self.F_mod = self.F_work.astype(np.float64)

    def _log_diagnostics(self):
        """Log detailed system diagnostics."""
        if not self.job_results_dir:
            return

        self.logger.debug("\n" + "="*40 + " System Diagnostics " + "="*40)
        
        # Matrix statistics
        orig_coo = self.K_orig.tocoo()
        mod_coo = self.K_mod.tocoo()
        
        stats = [
            f"Original Non-zeros: {orig_coo.nnz}",
            f"Modified Non-zeros: {mod_coo.nnz}",
            f"Penalty Value: {self.penalty:.2e}",
            f"Stabilization Factor: {1e-10 * self.penalty:.2e}"
        ]
        self.logger.debug("\n".join(stats))

        # Matrix content logging
        if mod_coo.shape[0] <= 100:
            df_K = pd.DataFrame(mod_coo.toarray(), dtype=np.float64)
            self.logger.debug("\nModified Stiffness Matrix:\n" + df_K.to_string(float_format="%.2e"))
        else:
            sparse_sample = pd.DataFrame({
                'Row': mod_coo.row,
                'Col': mod_coo.col,
                'Value': mod_coo.data
            }).sample(n=min(1000, len(mod_coo.data)))
            self.logger.debug("\nMatrix Sample (COO format):\n" + sparse_sample.to_string(index=False))

        # Condition number estimation
        diag_vals = mod_coo.diagonal()
        if np.any(diag_vals > 0):
            cond_estimate = diag_vals.max() / diag_vals[diag_vals > 0].min()
            self.logger.debug(f"\nCondition Estimate: {cond_estimate:.1e}")

    def save_modified_system(self, filename: str):
        """Save modified system to NPZ file."""
        if not filename.endswith('.npz'):
            filename += '.npz'
            
        np.savez_compressed(
            filename,
            K_data=self.K_mod.data,
            K_indices=self.K_mod.indices,
            K_indptr=self.K_mod.indptr,
            K_shape=self.K_mod.shape,
            F_global=self.F_mod,
            fixed_dofs=self.fixed_dofs
        )
        self.logger.info(f"💾 Saved modified system to {filename}")

    @classmethod
    def load_modified_system(cls, filename: str):
        """Load modified system from NPZ file."""
        data = np.load(filename)
        K = csr_matrix((
            data['K_data'],
            data['K_indices'],
            data['K_indptr']
        ), shape=data['K_shape'])
        return K, data['F_global'], data['fixed_dofs']
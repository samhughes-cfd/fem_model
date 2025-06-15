# processing_OOP\static\operations\boundary_conditions.py

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
import logging
from pathlib import Path
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
        self.job_results_dir = Path(job_results_dir) if job_results_dir else None
        self.K_mod: Optional[csr_matrix] = None
        self.F_mod: Optional[np.ndarray] = None
        self.penalty: Optional[float] = None

        raw_fixed = fixed_dofs if fixed_dofs is not None else range(6)
        self.fixed_dofs = np.array([int(dof) for dof in raw_fixed], dtype=int)

        self.logger = self._init_logging()
        self._validate_inputs()

    def _init_logging(self):
        logger = logging.getLogger(f"ModifyGlobalSystem.{id(self)}")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        log_path = None
        if self.job_results_dir:
            logs_dir = self.job_results_dir.parent / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            log_path = logs_dir / "ModifyGlobalSystem.log"
            try:
                file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
                file_handler.setFormatter(logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s "
                    "(Module: %(module)s, Line: %(lineno)d)"
                ))
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"⚠️ Failed to create file handler for ModifyGlobalSystem class log: {e}")

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(stream_handler)

        if log_path:
            logger.debug(f"📁 Log file created at: {log_path}")

        return logger

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
        self.penalty = np.float64(1e36) * (diag.max() if diag.size > 0 and diag.any() else 1e36)
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

    def _apply_stabilization(self, include_off_diagonal: bool = False):
        """Add numerical stabilization to matrix."""
        ε = np.float64(1e-10) * self.penalty
        diagonals = [ε]  # main diagonal
        offsets = [0]

        if include_off_diagonal:
            diagonals.extend([ε, ε])     # upper and lower
            offsets.extend([-1, 1])

        stabilization = diags(
            diagonals,
            offsets,
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
        """Save modified system to NPZ file with enforced float64 precision."""
        if not filename.endswith('.npz'):
            filename += '.npz'

        np.savez_compressed(
            filename,
            K_data=self.K_mod.data.astype(np.float64),
            K_indices=self.K_mod.indices.astype(np.int32),
            K_indptr=self.K_mod.indptr.astype(np.int32),
            K_shape=np.array(self.K_mod.shape, dtype=np.int32),
            F_global=self.F_mod.astype(np.float64),
            fixed_dofs=np.array(self.fixed_dofs, dtype=np.int32)
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
        fixed_dofs = [int(dof) for dof in data['fixed_dofs']]
        return K, data['F_global'], fixed_dofs


# processing_OOP\static\operations\modification.py

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import time
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix, diags
from typing import Sequence, Tuple, Optional, Union

FLOAT_FORMAT = "%.17e"        # keep full float64 precision

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
        local_global_dof_map: Sequence[np.ndarray],
        job_results_dir: Optional[str] = None,
        fixed_dofs: Optional[Union[Sequence[int], np.ndarray]] = None,
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

        self.local_global_dof_map: list[np.ndarray] = [
            np.asarray(a, dtype=np.int32) for a in local_global_dof_map
        ]


        raw_fixed = fixed_dofs if fixed_dofs is not None else range(6)
        # single, explicit conversion → contiguous C buffer
        self.fixed_dofs = np.asarray(raw_fixed, dtype=np.int32)

        self.logger = self._init_logging()
        self._validate_inputs()

        # After validation (we know F_orig exists and is valid size)
        self.free_dofs = np.setdiff1d(
            np.arange(self.F_orig.size, dtype=np.int32),
            self.fixed_dofs,
            assume_unique=True
            )

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
            #self._apply_stabilization()
            self._finalize_system()
            self._log_diagnostics()
            self._export_K_mod()
            self._export_F_mod()
            self._export_modify_local_global_dof_map()
        except Exception as e:
            self.logger.critical(f"❌ Boundary condition application failed: {str(e)}", exc_info=True)
            raise RuntimeError("Boundary condition application failed") from e
        
        exec_time = time.perf_counter() - start_time
        self.logger.info(f"✅ Boundary conditions applied in {exec_time:.2f}s")
        return self.K_mod, self.F_mod, self.fixed_dofs
    
    def _export_modify_local_global_dof_map(self):
        """Exports per-element local/global DOF map and fixed/free flags."""
        if self.job_results_dir is None:
            return

        maps_dir = self.job_results_dir.parent / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        path = maps_dir / "02_modification_map.csv"

        rows = []
        for eid, global_dofs in enumerate(self.local_global_dof_map):
            global_dofs = np.asarray(global_dofs, dtype=np.int32)  # Explicit coercion
            local_dofs = np.arange(len(global_dofs), dtype=np.int32)
            fixed_flags = np.isin(global_dofs, self.fixed_dofs).astype(int)

            rows.append({
                "Element ID": eid,
                "Local DOF": local_dofs.tolist(),
                "Global DOF": global_dofs.tolist(),
                "Fixed(1)/Free(0) Flag": fixed_flags.tolist()
            })

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        self.logger.info(f"📎 Element DOF fixed/free map saved to: {path}")

    def _convert_matrices(self):
        """Convert matrices to working formats with precision control."""
        self.K_work = self.K_orig.astype(np.float64).tolil()
        self.F_work = self.F_orig.copy().astype(np.float64)

    def _compute_penalty(self):
        """Calculate dynamic penalty value."""
        diag = self.K_work.diagonal()
        # safer penalty
        max_diag = np.max(np.abs(diag)) if diag.size else 1.0
        self.penalty = (1e36 if max_diag == 0 else 1e36 * max_diag)
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

    #def _apply_stabilization(self, include_off_diagonal: bool = False):
        #"""Add numerical stabilization to matrix."""
        #ε = np.float64(1e-10) * self.penalty
        #diagonals = [ε]  # main diagonal
        #offsets = [0]

        #if include_off_diagonal:
            #diagonals.extend([ε, ε])     # upper and lower
            #offsets.extend([-1, 1])

        #stabilization = diags(
            #diagonals,
            #offsets,
            #shape=self.K_work.shape,
            #format='lil',
            #dtype=np.float64
        #)

        #self.K_work = self.K_work + stabilization

    def _finalize_system(self):
        """Convert to final matrix formats."""
        self.K_mod = self.K_work.tocsr().astype(np.float64)
        self.F_mod = self.F_work.astype(np.float64)

    def _coo_to_dataframe(self, matrix: csr_matrix | coo_matrix, *,
                      value_label: str = "Value") -> pd.DataFrame:
        """Return a tidy DataFrame from a COO-format sparse matrix.

        Parameters
        ----------
        matrix : csr_matrix or coo_matrix
            Sparse matrix; will be converted to COO if necessary.
        value_label : str
            Column header for the numerical data.

        Returns
        -------
        pd.DataFrame
            Three columns: 'Row', 'Col', <value_label>.
        """
        if not isinstance(matrix, coo_matrix):
            matrix = matrix.tocoo()

        return pd.DataFrame({
            "Row":   matrix.row.tolist(),      # Python ints
            "Col":   matrix.col.tolist(),
            value_label: matrix.data           # float64
        })

    def _export_K_mod(self):
        if self.job_results_dir is None or self.K_mod is None:
            return

        path = self.job_results_dir / "03_K_mod.csv"
        df   = self._coo_to_dataframe(self.K_mod, value_label="K Value")
        df.to_csv(path, index=False, float_format=FLOAT_FORMAT)
        self.logger.info(f"💾 Modified stiffness matrix saved to: {path}")

    def _export_F_mod(self):
        """Write F_mod to <primary_results>/F_mod.csv."""
        if self.job_results_dir is None or self.F_mod is None:
            return
        path = self.job_results_dir / "04_F_mod.csv"
        df   = pd.DataFrame({
            "DOF":   list(range(self.F_mod.size)),   # python int
            "F Value": self.F_mod                    # float64
        })
        df.to_csv(path, index=False, float_format=FLOAT_FORMAT)
        self.logger.info(f"💾 Modified force vector saved to: {path}")

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
            sparse_sample = (self._coo_to_dataframe(mod_coo).sample(n=min(1000, mod_coo.nnz), random_state=0))

            self.logger.debug("\nMatrix Sample (COO format):\n" + sparse_sample.to_string(index=False))

        # Condition number estimation
        diag_vals = mod_coo.diagonal()
        if np.any(diag_vals > 0):
            cond_estimate = diag_vals.max() / diag_vals[diag_vals > 0].min()
            self.logger.debug(f"\nCondition Estimate: {cond_estimate:.1e}")
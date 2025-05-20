# processing_OOP\static\operations\condensation.py

import os
import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from typing import Tuple, Optional, Dict
import time

class CondenseGlobalSystem:
    """Static condensation system with advanced validation and adaptive numerics."""
    
    def __init__(
        self,
        K_mod: csr_matrix,
        F_mod: np.ndarray,
        fixed_dofs: np.ndarray,
        job_results_dir: Optional[str] = None,
        base_tol: float = 1e-12
    ):
        self.K_mod = K_mod.astype(np.float64, copy=False)
        self.F_mod = F_mod.astype(np.float64, copy=False)
        self.fixed_dofs = fixed_dofs
        self.job_results_dir = job_results_dir
        self.base_tol = float(base_tol)
        self.condensed_dofs = None
        self.inactive_dofs = None
        self.K_cond = None
        self.F_cond = None
        self.mapping = {}
        self.adaptive_tol = base_tol
        self._init_logging()
        self._validate_system()

    def _init_logging(self):
        """Configure hierarchical logging with validation tracking."""
        self.logger = logging.getLogger(f"EnhancedCondensation.{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        if self.job_results_dir:
            os.makedirs(self.job_results_dir, exist_ok=True)
            log_path = os.path.join(self.job_results_dir, "condensation.log")
            file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] (%(funcName)s:%(lineno)d) - %(message)s"
            ))
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.CRITICAL)
        self.logger.addHandler(console_handler)

    def _validate_system(self):
        """Comprehensive system validation before processing."""
        self._validate_indices()
        self._validate_matrix_properties()
        self._compute_adaptive_tolerance()
        
        self.logger.debug(
            "System Validation Summary:\n"
            f"  - Fixed DOFs Valid: {len(self.fixed_dofs)}/{self.K_mod.shape[0]}\n"
            f"  - Matrix Shape: {self.K_mod.shape}\n"
            f"  - Adaptive Tolerance: {self.adaptive_tol:.2e}"
        )

    def _validate_indices(self):
        """Strict index validation with error feedback."""
        if not isinstance(self.fixed_dofs, np.ndarray):
            raise TypeError("Fixed DOFs must be numpy array")
            
        if not np.issubdtype(self.fixed_dofs.dtype, np.integer):
            raise TypeError("Fixed DOFs must be integer indices")
            
        if np.any(self.fixed_dofs < 0):
            invalid = self.fixed_dofs[self.fixed_dofs < 0]
            raise ValueError(f"Negative DOF indices: {invalid}")
            
        if len(np.unique(self.fixed_dofs)) != len(self.fixed_dofs):
            duplicates = self.fixed_dofs[np.diff(np.sort(self.fixed_dofs)) == 0]
            raise ValueError(f"Duplicate fixed DOFs: {np.unique(duplicates)}")
            
        if np.max(self.fixed_dofs) >= self.K_mod.shape[0]:
            invalid = self.fixed_dofs[self.fixed_dofs >= self.K_mod.shape[0]]
            raise ValueError(f"Fixed DOFs exceed matrix dimension: {invalid}")

    def _validate_matrix_properties(self):
        """Matrix integrity checks."""
        if not issparse(self.K_mod):
            raise TypeError("Stiffness matrix must be sparse")
            
        if self.K_mod.shape[0] != self.K_mod.shape[1]:
            raise ValueError("Non-square stiffness matrix")
            
        if self.K_mod.shape[0] != self.F_mod.shape[0]:
            raise ValueError("Matrix/vector dimension mismatch")

    def _compute_adaptive_tolerance(self):
        """Auto-scale tolerance to matrix magnitude."""
        if self.K_mod.nnz == 0:
            self.adaptive_tol = self.base_tol
            return
            
        max_val = max(np.abs(self.K_mod.data).max(), np.abs(self.F_mod).max())
        self.adaptive_tol = max(self.base_tol, 1e-12 * max_val)
        self.logger.debug(f"Adaptive tolerance: {self.adaptive_tol:.2e}")

    def apply_condensation(self) -> Tuple[np.ndarray, np.ndarray, csr_matrix, np.ndarray]:
        """Robust condensation process with validation checkpoints."""
        start_time = time.perf_counter()
        self.logger.info("🚀 Starting enhanced condensation process")
        
        try:
            self._compute_active_dofs()
            self._create_intermediate_system()
            self._identify_fully_active_dofs()
            self._validate_condensation()
            self._build_condensed_system()
            self._create_verified_mapping()
            self._log_system_details()
        except Exception as e:
            self.logger.critical(f"❌ Condensation failed: {str(e)}", exc_info=True)
            raise RuntimeError("Condensation aborted") from e
            
        exec_time = time.perf_counter() - start_time
        self.logger.info(f"✅ Condensation completed in {exec_time:.2f}s")
        return self.condensed_dofs, self.inactive_dofs, self.K_cond, self.F_cond

    def _compute_active_dofs(self):
        """Compute active DOFs with validation."""
        self.active_dofs = np.setdiff1d(
            np.arange(self.K_mod.shape[0]), 
            self.fixed_dofs,
            assume_unique=True
        )
        if len(self.active_dofs) == 0:
            raise ValueError("No active DOFs remaining after fixed DOF removal")
            
        self.logger.debug(f"Active DOFs: {self._format_dof_sample(self.active_dofs)}")

    def _create_intermediate_system(self):
        """Create intermediate system with sparse-safe operations."""
        self.K_intermediate = self.K_mod[self.active_dofs][:, self.active_dofs].tolil()
        self.F_intermediate = self.F_mod[self.active_dofs]
        self.logger.debug(
            f"Intermediate system created | Shape: {self.K_intermediate.shape}"
        )

    def _identify_fully_active_dofs(self):
        """Identify non-zero DOFs using sparse-safe methods."""
        # Sparse row-wise non-zero detection
        nonzero_rows = np.unique(self.K_intermediate.nonzero()[0])
        self.condensed_dofs = self.active_dofs[nonzero_rows]
        self.inactive_dofs = np.setdiff1d(self.active_dofs, self.condensed_dofs)
        
        self.logger.info(
            f"Secondary condensation removed {len(self.inactive_dofs)} DOFs"
        )
        if len(self.condensed_dofs) == 0:
            raise ValueError("All active DOFs removed in secondary condensation")

    def _validate_condensation(self):
        """Post-condensation validation checks."""
        # Check for fixed DOF contamination
        overlap = np.intersect1d(self.condensed_dofs, self.fixed_dofs, assume_unique=True)
        if overlap.size > 0:
            raise ValueError(f"Fixed DOFs in condensed set: {overlap}")
            
        # Check index bounds
        if np.max(self.condensed_dofs) >= self.K_mod.shape[0]:
            invalid = self.condensed_dofs[self.condensed_dofs >= self.K_mod.shape[0]]
            raise ValueError(f"Invalid condensed DOFs: {invalid}")

    def _create_verified_mapping(self):
        """Create and validate bi-directional DOF mapping."""
        # Forward mapping
        self.condensed_to_original = {
            c_idx: o_idx 
            for c_idx, o_idx in enumerate(self.condensed_dofs)
        }
        
        # Reverse mapping
        self.original_to_condensed = {
            o_idx: c_idx 
            for c_idx, o_idx in self.condensed_to_original.items()
        }
        
        # Verify completeness
        missing = set(self.condensed_dofs) - set(self.original_to_condensed.keys())
        if missing:
            raise ValueError(f"Missing reverse mapping for: {missing}")
            
        self.logger.debug("Bi-directional mapping validated")

    def _log_system_details(self):
        """Comprehensive system logging with mapping integrity."""
        self.logger.debug("\n" + "="*40 + " SYSTEM DETAILS " + "="*40)
        
        # Mapping statistics
        total_dofs = self.K_mod.shape[0]
        stats = [
            f"Total DOFs: {total_dofs}",
            f"Fixed DOFs: {len(self.fixed_dofs)} ({len(self.fixed_dofs)/total_dofs:.1%})",
            f"Active DOFs: {len(self.active_dofs)} ({len(self.active_dofs)/total_dofs:.1%})",
            f"Condensed DOFs: {len(self.condensed_dofs)} ({len(self.condensed_dofs)/total_dofs:.1%})",
            f"Inactive DOFs: {len(self.inactive_dofs)} ({len(self.inactive_dofs)/total_dofs:.1%})"
        ]
        self.logger.debug("📊 Statistics:\n" + "\n".join(stats))
        
        # Sample mappings
        self.logger.debug(
            "🗺️ Mapping Samples:\n"
            f"Condensed → Original: {self._format_mapping_sample(self.condensed_to_original)}\n"
            f"Original → Condensed: {self._format_mapping_sample(self.original_to_condensed)}"
        )

        # Matrix diagnostics
        if len(self.condensed_dofs) <= 100:
            self._log_full_matrices()
        else:
            self._log_sparse_pattern()

    def _format_mapping_sample(self, mapping, n=5):
        """Format mapping samples for readability."""
        items = list(mapping.items())
        sample = items[:n] + [("...", "...")] + items[-n:] if len(items) > 2*n else items
        return "\n".join(f"{k} → {v}" for k,v in sample)

    def _log_full_matrices(self):
        """Detailed matrix logging for small systems."""
        K_df = pd.DataFrame(
            self.K_cond.toarray(),
            index=[f"C{i} (O{self.condensed_to_original[i]})" for i in range(len(self.condensed_dofs))],
            columns=[f"C{j} (O{self.condensed_to_original[j]})" for j in range(len(self.condensed_dofs))]
        )
        self.logger.debug(f"🔍 Condensed Stiffness Matrix:\n{K_df.to_string(float_format='%.2e')}")
        
        F_df = pd.DataFrame(
            self.F_cond,
            index=[f"C{i} (O{self.condensed_to_original[i]})" for i in range(len(self.condensed_dofs))],
            columns=["Force"]
        )
        self.logger.debug(f"🔍 Condensed Force Vector:\n{F_df.to_string(float_format='%.2e')}")

    def _log_sparse_pattern(self):
        """Efficient sparse pattern logging for large systems."""
        coo = self.K_cond.tocoo()
        sample = pd.DataFrame({
            'Row': coo.row,
            'Col': coo.col,
            'Value': coo.data
        }).sample(n=min(1000, len(coo.data)), random_state=42)
        
        self.logger.debug(
            "🔍 Sparse Matrix Pattern Sample:\n" + 
            sample.to_string(index=False, float_format="%.2e")
        )

    def save_condensed_system(self, filename: str):
        """Save condensed system with full metadata."""
        if not filename.endswith('.npz'):
            filename += '.npz'
            
        np.savez_compressed(
            filename,
            K_cond_data=self.K_cond.data,
            K_cond_indices=self.K_cond.indices,
            K_cond_indptr=self.K_cond.indptr,
            K_cond_shape=self.K_cond.shape,
            F_cond=self.F_cond,
            condensed_dofs=self.condensed_dofs,
            inactive_dofs=self.inactive_dofs,
            adaptive_tol=self.adaptive_tol,
            fixed_dofs=self.fixed_dofs
        )
        self.logger.info(f"💾 Saved condensed system to {filename}")

    @classmethod
    def load_condensed_system(cls, filename: str):
        """Load condensed system with metadata validation."""
        data = np.load(filename)
        K_cond = csr_matrix((
            data['K_cond_data'],
            data['K_cond_indices'],
            data['K_cond_indptr']
        ), shape=data['K_cond_shape'])
        return (
            data['condensed_dofs'],
            data['inactive_dofs'],
            K_cond,
            data['F_cond']
        )
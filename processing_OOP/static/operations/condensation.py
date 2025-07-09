# processing_OOP\static\operations\condensation.py

import os
import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix, issparse   # add coo_matrix
from typing import Sequence, Union, Tuple, Optional, Dict   # Dict gets used below
from pathlib import Path
from processing_OOP.static.results.containers.map_results import MapEntry

class CondenseModifiedSystem:
    """Static condensation system with advanced validation and adaptive numerics."""

    def __init__(
        self,
        K_mod: csr_matrix,
        F_mod: np.ndarray,
        fixed_dofs: Union[Sequence[int], np.ndarray],
        local_global_dof_map: list[np.ndarray],
        job_results_dir: Optional[str] = None,
        base_tol: float = 1e-12
    ):
        self.K_mod = K_mod.astype(np.float64, copy=False)
        self.F_mod = F_mod.astype(np.float64, copy=False)
        self.fixed_dofs = np.array(fixed_dofs, dtype=np.int32)
        self.local_global_dof_map = local_global_dof_map
        self.job_results_dir = Path(job_results_dir) if job_results_dir else None
        self.base_tol = float(base_tol)
        self.condensed_dofs = None
        self.inactive_dofs = None
        self.K_cond = None
        self.F_cond = None
        self.mapping = {}
        self.adaptive_tol = base_tol
        self.logger = self._init_logging()
        self._validate_system()

    def _init_logging(self):
        logger = logging.getLogger(f"CondenseModifiedSystem.{id(self)}")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        log_path = None
        if self.job_results_dir:
            logs_dir = self.job_results_dir.parent / "logs"  # ✅ Store logs alongside primary_results
            logs_dir.mkdir(parents=True, exist_ok=True)

            log_path = logs_dir / "CondenseModifiedSystem.log"
            try:
                file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
                file_handler.setFormatter(logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s "
                    "(Module: %(module)s, Line: %(lineno)d)"
                ))
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"⚠️ Failed to create file handler for CondenseModifiedSystem class log: {e}")

        # Console output (INFO level and above)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(stream_handler)

        if log_path:
            logger.debug(f"📁 Log file created at: {log_path}")

        return logger

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

    def apply_condensation(self):
        """Apply static condensation, preserving curvature DOFs and pruning only structurally zeroed DOFs."""
        self._compute_active_dofs()
        self.condensed_dofs = self.active_dofs.copy()
        self.inactive_dofs = np.array([], dtype=int)
        self._create_intermediate_system()

        if self._has_truly_singular_dofs():
            self.logger.warning("⚠️ Detected DOFs with zero stiffness coupling — pruning these DOFs.")
            self._prune_truly_singular_dofs()

        self._validate_condensation()
        self._build_condensed_system()
        self._create_verified_mapping()
        self._export_condensed_map_legacy()
        self._build_condensation_map()
        self._export_K_cond()
        self._export_F_cond()
        self._log_system_details()

        self.condensation_map = self._build_condensation_map()

        return self.condensed_dofs, self.inactive_dofs, self.K_cond, self.F_cond, self.condensation_map


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
        mask = np.abs(self.K_intermediate).sum(axis=1).A1 > self.adaptive_tol
        nonzero_rows = np.where(mask)[0]
        self.condensed_dofs = self.active_dofs[nonzero_rows]
        self.inactive_dofs = np.setdiff1d(self.active_dofs, self.condensed_dofs)
        
        self.logger.info(
            f"Secondary condensation removed {len(self.inactive_dofs)} DOFs"
        )
        if len(self.condensed_dofs) == 0:
            raise ValueError("All active DOFs removed in secondary condensation")
        
    def _build_condensed_system(self):
        """Extract condensed system matrices using validated DOF subset."""
        self.logger.debug("🔧 Building condensed system from intermediate matrix")
    
        # Extract condensed stiffness matrix
        K_c = self.K_intermediate[
            np.isin(self.active_dofs, self.condensed_dofs)
        ][:, np.isin(self.active_dofs, self.condensed_dofs)].tocsr()
    
        # Extract corresponding condensed force vector
        F_c = self.F_intermediate[np.isin(self.active_dofs, self.condensed_dofs)]

        # Final assignments
        self.K_cond = K_c
        self.F_cond = F_c

        self.logger.debug(
            f"📐 Condensed system built: "
            f"K_cond shape = {self.K_cond.shape}, "
            f"F_cond length = {self.F_cond.shape[0]}"
        )

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
    
    def _format_dof_sample(self, dofs, n=10):
        """Format a sample of DOFs for concise logging."""
        dofs = np.asarray(dofs)
        if len(dofs) == 0:
            return "[]"
        sample = dofs[:n]
        tail = "..." if len(dofs) > n else ""
        return "[" + ", ".join(map(str, sample)) + tail + "]"
    
    # ------------------------------------------------------------------ utils
    @staticmethod
    def _coo_to_dataframe(mat: csr_matrix | coo_matrix,
                          value_label: str = "Value") -> pd.DataFrame:
        """Return a tidy Δ-frame of COO data (Row, Col, Value)."""
        if not isinstance(mat, coo_matrix):
            mat = mat.tocoo()
        return pd.DataFrame({
            "Row":   mat.row.astype(int),
            "Col":   mat.col.astype(int),
            value_label: mat.data
        })

    def _export_K_cond(self):
        if self.job_results_dir is None or self.K_cond is None:
            return
        path = self.job_results_dir / "05_K_cond.csv"
        #df   = self._coo_to_dataframe(self.K_cond, value_label="K Value")
        #df.to_csv(path, index=False, float_format="%.17e")
        #self.logger.info(f"💾 Condensed stiffness matrix saved → {path}")

    def _export_F_cond(self):
        if self.job_results_dir is None or self.F_cond is None:
            return
        path = self.job_results_dir / "06_F_cond.csv"
        #df   = pd.DataFrame({
                  #"DOF":   np.arange(self.F_cond.size, dtype=int),
                  #"F Value": self.F_cond
              #})
        #df.to_csv(path, index=False, float_format="%.17e")
        #self.logger.info(f"💾 Condensed force vector   saved → {path}")

    def _export_condensed_map_legacy(self):
        """
        Per-element condensation mapping:
        Element ID,Local DOF,Global DOF,
        Fixed(1)/Free(0) Flag,Zero(1)/Non-zero(0) Flag,
        Active(1)/Inactive(0) Flag,Condensed DOF
        """
        # ------------------------------------------------------------------ guard
        if (
            self.job_results_dir is None
            or not hasattr(self, "original_to_condensed")
            or not hasattr(self, "local_global_dof_map")  # <-- MUST be set by caller
        ):
            return

        maps_dir = self.job_results_dir.parent / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        path = maps_dir / "03_condensation_map.csv"

        # ------------------------------------------------------------------ helpers
        def flags_for(dof_list):
            fixed   = [1 if d in self.fixed_dofs else 0 for d in dof_list]
            zero    = [1 if (d not in self.fixed_dofs and d not in self.condensed_dofs) else 0 for d in dof_list]
            active  = [1 if d in self.condensed_dofs  else 0 for d in dof_list]
            cond    = [self.original_to_condensed.get(d, "") for d in dof_list]
            return fixed, zero, active, cond

        # ------------------------------------------------------------------ rows
        rows = []
        for elem_id, g_dofs in enumerate(self.local_global_dof_map):
            l_dofs = list(range(len(g_dofs)))             # 0…11 etc.
            fixed, zero, active, cond = flags_for(g_dofs)

            rows.append({
                "Element ID": elem_id,
                "Local DOF":               str(l_dofs),
                "Global DOF":              str(g_dofs.tolist()),
                "Fixed(1)/Free(0) Flag":   str(fixed),
                "Zero(1)/Non-zero(0) Flag":str(zero),
                "Active(1)/Inactive(0) Flag": str(active),
                "Condensed DOF":           str(cond),
            })

        # ------------------------------------------------------------------ write
        #pd.DataFrame(rows).to_csv(path, index=False)
        #self.logger.info(f"🗺️  Structured condensation map saved → {path}")

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

    def _has_truly_singular_dofs(self) -> bool:
        """Check for DOFs with zero stiffness coupling (rows AND columns of all zeros)."""
        submatrix = self.K_mod[self.active_dofs][:, self.active_dofs].tocsr()
        row_nnz = np.diff(submatrix.indptr)
        col_nnz = np.diff(submatrix.tocsc().indptr)
        zero_rows = row_nnz == 0
        zero_cols = col_nnz == 0
        return np.any(zero_rows & zero_cols)

    def _prune_truly_singular_dofs(self):
        """Prune only DOFs with no structural participation (zero in both rows and columns)."""
        submatrix = self.K_mod[self.active_dofs][:, self.active_dofs].tocsr()
        row_nnz = np.diff(submatrix.indptr)
        col_nnz = np.diff(submatrix.tocsc().indptr)
        structurally_active = (row_nnz > 0) & (col_nnz > 0)

        keep_indices = np.where(structurally_active)[0]
        self.condensed_dofs = self.active_dofs[keep_indices]
        self.inactive_dofs = np.setdiff1d(self.active_dofs, self.condensed_dofs)

        if len(self.condensed_dofs) == 0:
            raise ValueError("All DOFs removed — system appears fully disconnected. Check K_mod and BCs.")

        self.logger.warning(
            f"Pruned {len(self.inactive_dofs)} DOFs with no structural connectivity from the active set. "
            f"{len(self.condensed_dofs)} DOFs retained for condensation."
        )
    
    def _build_condensation_map(self) -> list[MapEntry]:
        """
        Construct detailed DOF mapping after condensation.
        Captures fixed, zeroed, active, and condensed indices.
        """
        condensation_map = []
        for i, global_dofs in enumerate(self.local_global_dof_map):
            entry = MapEntry(
                element_id=i,
                local_dof=np.arange(len(global_dofs), dtype=np.int32),
                global_dof=global_dofs,
                fixed_flag=np.array(
                    [1 if d in self.fixed_dofs else 0 for d in global_dofs], dtype=np.int32
                ),
                zero_flag=np.array(
                    [1 if (d not in self.fixed_dofs and d not in self.condensed_dofs) else 0 for d in global_dofs],
                    dtype=np.int32
                ),
                active_flag=np.array(
                    [1 if d in self.condensed_dofs else 0 for d in global_dofs], dtype=np.int32
                ),
                condensed_dof=np.array(
                    [self.original_to_condensed.get(d, -1) for d in global_dofs], dtype=np.int32
                )
            )
            condensation_map.append(entry)

        self.condensation_map = condensation_map
        return condensation_map


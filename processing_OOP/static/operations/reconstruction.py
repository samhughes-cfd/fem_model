# processing_OOP/static/operations/reconstruction.py

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Sequence
import matplotlib.pyplot as plt
import time
from datetime import datetime
from processing_OOP.static.results.containers.map_results import MapEntry

class ReconstructGlobalSystem:
    """High-performance displacement reconstruction system with validation and diagnostics."""

    def __init__(
        self,
        active_dofs: np.ndarray,
        U_cond: np.ndarray,
        total_dofs: int,
        job_results_dir: Path,
        *,
        local_global_dof_map: Sequence[np.ndarray],   # NEW
        fixed_dofs: Optional[np.ndarray] = None,
        inactive_dofs: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        active_dofs : np.ndarray
            Array of active DOF indices (1D int array)
        U_cond : np.ndarray
            Condensed displacement solution vector
        total_dofs : int
            Total degrees of freedom in the global system
        job_results_dir : Path
            Directory for reconstruction logs and outputs
        fixed_dofs : Optional[np.ndarray], optional
            Array of fixed DOF indices for validation
        inactive_dofs : Optional[np.ndarray], optional
            Array of inactive DOFs used during condensation (for validation)
        """
        self.active_dofs: np.ndarray = active_dofs.astype(np.int64)
        self.U_cond: np.ndarray = U_cond.astype(np.float64)
        self.total_dofs: int = int(total_dofs)
        self.job_results_dir: Path = Path(job_results_dir)
        self.fixed_dofs: np.ndarray = fixed_dofs if fixed_dofs is not None else np.array([], dtype=np.int32)
        self.inactive_dofs: Optional[np.ndarray] = inactive_dofs
        self.local_global_dof_map = list(local_global_dof_map)

        self.U_global: np.ndarray = np.zeros(self.total_dofs, dtype=np.float64)
        self.reconstruction_time: Optional[float] = None
        self.logger: logging.Logger = self._init_logging()

        # Add the lookup mapping here
        self.original_to_condensed = {g: c for c, g in enumerate(self.active_dofs)}

        self._validate_inputs()


    def _init_logging(self) -> logging.Logger:
        """Initialize logging infrastructure for diagnostics."""
        logger = logging.getLogger(f"ReconstructGlobalSystem.{id(self)}")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        log_path = None
        if self.job_results_dir:
            logs_dir = self.job_results_dir.parent / "logs"  # ✅ Sibling to primary_results
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / "ReconstructGlobalSystem.log"

            try:
                file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
                file_handler.setFormatter(logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s (Module: %(module)s, Line: %(lineno)d)"
                ))
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"⚠️ Failed to create file handler for ReconstructGlobalSystem class log: {e}")

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(stream_handler)

        if log_path:
            logger.debug(f"📁 Log file created at: {log_path}")

        return logger

    def _validate_inputs(self) -> None:
        """Comprehensive input validation with error aggregation."""
        errors = []

        if not isinstance(self.active_dofs, np.ndarray):
            errors.append("active_dofs must be a numpy array")
        if not isinstance(self.U_cond, np.ndarray):
            errors.append("U_cond must be a numpy array")
        if not isinstance(self.total_dofs, int):
            errors.append("total_dofs must be an integer")
        if not isinstance(self.fixed_dofs, np.ndarray):
            errors.append("fixed_dofs must be a numpy array")

        if self.active_dofs.dtype != np.int64:
            errors.append("active_dofs must be of dtype int64")
        if self.U_cond.dtype != np.float64:
            errors.append("U_cond must be of dtype float64")
        if self.fixed_dofs.size > 0 and self.fixed_dofs.dtype != np.int32:
            errors.append("fixed_dofs must be of dtype int32 if provided")

        if self.active_dofs.ndim != 1:
            errors.append(f"active_dofs must be 1D array, got shape {self.active_dofs.shape}")
        if self.U_cond.ndim != 1:
            errors.append(f"U_cond must be 1D array, got shape {self.U_cond.shape}")
        if self.fixed_dofs.ndim != 1:
            errors.append(f"fixed_dofs must be 1D array, got shape {self.fixed_dofs.shape}")

        if self.total_dofs <= 0:
            errors.append(f"total_dofs must be positive, got {self.total_dofs}")
        if len(self.U_cond) != len(self.active_dofs):
            errors.append(
                f"Length mismatch: U_cond ({len(self.U_cond)}) vs active_dofs ({len(self.active_dofs)})"
            )
        if np.any(self.active_dofs >= self.total_dofs):
            invalid = self.active_dofs[self.active_dofs >= self.total_dofs]
            errors.append(f"Active DOFs exceed total DOFs: {invalid}")
        if np.any(self.active_dofs < 0):
            invalid = self.active_dofs[self.active_dofs < 0]
            errors.append(f"Negative active DOFs: {invalid}")
        if self.fixed_dofs.size > 0 and (np.any(self.fixed_dofs >= self.total_dofs) or np.any(self.fixed_dofs < 0)):
            invalid = self.fixed_dofs[
                (self.fixed_dofs >= self.total_dofs) | (self.fixed_dofs < 0)
            ]
            errors.append(f"Invalid fixed DOF indices (out of bounds): {invalid}")

        if errors:
            error_msg = "Input validation failed:\n  " + "\n  ".join(errors)
            self.logger.critical(error_msg)
            raise ValueError(error_msg)
        else:
            self.logger.debug("✅ Input validation passed")

    def reconstruct(self) -> np.ndarray:
        """Execute full reconstruction pipeline with diagnostics.

        Returns
        -------
        np.ndarray
            Reconstructed global displacement vector.
        """
        start_time = time.perf_counter()
        self.logger.info("🚀 Starting displacement reconstruction")

        try:
            self._perform_mapping()
            self._validate_reconstruction()
            self._export_reconstruction_map()
            self._log_statistics()
            self._save_results()
            self._build_reconstruction_map()
        except Exception as e:
            self.logger.critical(f"❌ Reconstruction failed: {str(e)}", exc_info=True)
            raise RuntimeError("Displacement reconstruction failed") from e

        self.reconstruction_time = time.perf_counter() - start_time
        self.logger.info(f"✅ Reconstruction completed in {self.reconstruction_time:.2f}s")


        self.reconstruction_map = self._build_reconstruction_map()
        
        return self.U_global, self.reconstruction_map

    def _perform_mapping(self) -> None:
        """Vectorized mapping of condensed displacements to global system."""
        self.U_global[self.active_dofs] = self.U_cond

        # Add this line:
        self.original_to_condensed = {g: c for c, g in enumerate(self.active_dofs)}

        if self.fixed_dofs.size > 0:
            fixed_nonzero = np.nonzero(self.U_global[self.fixed_dofs])[0]
            if fixed_nonzero.size > 0:
                self.logger.warning(
                    f"Non-zero displacements at fixed DOFs: {self.fixed_dofs[fixed_nonzero]}"
                )


    # --------------------------------------------------------------------- NEW
    def _export_reconstruction_map(self) -> None:
        """
        Write 04_reconstruction_map.csv in <primary_results>/../maps, one row per
        element with list-style payloads (to stay consistent with maps 01-03).
        """
        if self.job_results_dir is None:
            return                                     # nothing to do

        maps_dir = self.job_results_dir.parent / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        path = maps_dir / "04_reconstruction_map.csv"

        # ---------- fast, vectorised flag arrays over *all* global DOFs ----------
        all_dofs      = np.arange(self.total_dofs, dtype=np.int32)
        fixed_mask    = np.isin(all_dofs, self.fixed_dofs, assume_unique=True)
        inactive_mask = np.isin(all_dofs, self.inactive_dofs if self.inactive_dofs is not None else [], assume_unique=True)
        active_mask   = np.isin(all_dofs, self.active_dofs,   assume_unique=True)

        # condensed index lookup  (-1 → not active)
        orig_to_cond = {g: c for c, g in enumerate(self.active_dofs)}
        condensed_idx = np.fromiter((orig_to_cond.get(d, -1) for d in all_dofs),
                                dtype=np.int32, count=self.total_dofs)

        # ---------- per-element rows ----------
        rows = []
        for elem_id, g_dofs in enumerate(self.local_global_dof_map):
            g_dofs = np.asarray(g_dofs, dtype=np.int32)
            l_dofs = np.arange(g_dofs.size, dtype=np.int32)

            rows.append({
                "Element ID": elem_id,                         # still loop index
                "Local DOF":                str(l_dofs.tolist()),
                "Global DOF":               str(g_dofs.tolist()),
                "Fixed(1)/Free(0) Flag":    str(fixed_mask[g_dofs].astype(int).tolist()),
                "Zero(1)/Non-zero(0) Flag": str(inactive_mask[g_dofs].astype(int).tolist()),
                "Active(1)/Inactive(0) Flag": str(active_mask[g_dofs].astype(int).tolist()),
                "Condensed DOF":            str(condensed_idx[g_dofs].tolist()),
                "Reconstructed Global DOF": str(self.U_global[g_dofs].tolist())
            })

        #pd.DataFrame(rows).to_csv(path, index=False, float_format="%.17e")
        #self.logger.info(f"🗺️  Reconstruction map saved → {path}")

    def _validate_reconstruction(self) -> None:
        """Quality checks on reconstructed solution."""
        nan_count = np.isnan(self.U_global).sum()
        if nan_count > 0:
            raise ValueError(f"{nan_count} NaN values in reconstructed displacements")

        active_energy = np.dot(self.U_cond, self.U_cond)
        global_energy = np.dot(self.U_global, self.U_global)
        energy_diff = abs(active_energy - global_energy)

        if energy_diff > 1e-12 * max(active_energy, global_energy):
            self.logger.warning(
                f"Energy discrepancy detected: {energy_diff:.2e} "
                f"(Global: {global_energy:.2e}, Active: {active_energy:.2e})"
            )

    def _log_statistics(self) -> None:
        """Log detailed reconstruction statistics."""
        stats = [
            f"Total DOFs: {self.total_dofs}",
            f"Active DOFs: {len(self.active_dofs)}",
            f"Fixed DOFs: {len(self.fixed_dofs)}",
            f"Min displacement: {np.min(self.U_global):.3e}",
            f"Max displacement: {np.max(self.U_global):.3e}",
            f"Mean absolute displacement: {np.mean(np.abs(self.U_global)):.3e}"
        ]
        self.logger.info("📊 Reconstruction Statistics:\n  " + "\n  ".join(stats))

    def _save_results(self) -> None:
        """Write 08_U_global.csv in the same style as 07_U_cond.csv."""
        path = self.job_results_dir / "08_U_global.csv"
        #pd.DataFrame({
            #"Global DOF":    np.arange(self.U_global.size, dtype=int),
            #"U Value": self.U_global
        #}).to_csv(path, index=False, float_format="%.17e")
        #self.logger.info(f"💾 Global displacement saved → {path}")

    @property
    def solution(self) -> np.ndarray:
        """Get reconstructed displacement vector with copy protection.

        Returns
        -------
        np.ndarray
            Reconstructed displacement vector (copy).
        """
        return self.U_global.copy()

    def get_displacement(self, dof: int) -> float:
        """Safe accessor for individual DOF displacements.

        Parameters
        ----------
        dof : int
            Degree of freedom index to retrieve.

        Returns
        -------
        float
            Displacement value for the given DOF.

        Raises
        ------
        ValueError
            If index is out of bounds.
        """
        if not 0 <= dof < self.total_dofs:
            raise ValueError(f"Invalid DOF index: {dof}")
        return self.U_global[dof]

    def _build_reconstruction_map(self) -> list[MapEntry]:
        """
        Construct detailed DOF mapping after reconstruction.
        Includes fixed, zeroed, active, condensed, and reconstructed values.
        """
        return [
            MapEntry(
                element_id=i,
                local_dof=np.arange(len(global_dofs), dtype=np.int32),
                global_dof=global_dofs,
                fixed_flag=np.array(
                    [1 if d in self.fixed_dofs else 0 for d in global_dofs],
                    dtype=np.int32
                ),
                zero_flag=np.array(
                    [1 if (d not in self.fixed_dofs and d not in self.active_dofs) else 0 for d in global_dofs],
                    dtype=np.int32
                ),
                active_flag=np.array(
                    [1 if d in self.active_dofs else 0 for d in global_dofs],
                    dtype=np.int32
                ),
                condensed_dof=np.array(
                    [self.original_to_condensed.get(d, -1) for d in global_dofs],
                    dtype=np.int32
                ),
                reconstructed_values=np.array(
                    [self.U_global[d] for d in global_dofs],
                    dtype=np.float64
                )
            )
            for i, global_dofs in enumerate(self.local_global_dof_map)
        ]

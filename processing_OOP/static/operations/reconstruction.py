# processing_OOP\static\operations\reconstruction.py

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt

class ReconstructGlobalSystem:
    """High-performance displacement reconstruction system with validation and diagnostics."""
    
    def __init__(
        self,
        active_dofs: np.ndarray,
        U_cond: np.ndarray,
        total_dofs: int,
        job_results_dir: Path,
        fixed_dofs: Optional[np.ndarray] = None
    ):
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
        fixed_dofs : Optional[np.ndarray]
            Array of fixed DOF indices for validation
        """
        self.active_dofs = active_dofs.astype(np.int64)
        self.U_cond = U_cond.astype(np.float64)
        self.total_dofs = int(total_dofs)
        self.job_results_dir = Path(job_results_dir)
        self.fixed_dofs = fixed_dofs if fixed_dofs is not None else np.array([], dtype=np.int64)
        
        self.U_global = np.zeros(self.total_dofs, dtype=np.float64)
        self.reconstruction_time = None
        self._configure_logging()
        self._validate_inputs()

    def _configure_logging(self):
        """Initialize reconstruction-specific logging."""
        self.job_results_dir.mkdir(exist_ok=True, parents=True)
        
        self.logger = logging.getLogger(f"ReconstructionSystem.{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # File handler for structured logging
        log_file = self.job_results_dir / "reconstruction.log"
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        ))
        self.logger.addHandler(file_handler)

        # Console handler for warnings only
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        self.logger.addHandler(console_handler)

    def _validate_inputs(self):
        """Comprehensive input validation with error aggregation."""
        errors = []
        
        # Type checks
        if not isinstance(self.active_dofs, np.ndarray):
            errors.append("active_dofs must be a numpy array")
            
        if self.active_dofs.dtype != np.int64:
            errors.append("active_dofs must be integer indices")
            
        # Dimension checks
        if len(self.U_cond) != len(self.active_dofs):
            errors.append(
                f"U_cond length ({len(self.U_cond)}) "
                f"≠ active_dofs count ({len(self.active_dofs)})"
            )
            
        # Bounds checks
        if np.any(self.active_dofs >= self.total_dofs):
            invalid = self.active_dofs[self.active_dofs >= self.total_dofs]
            errors.append(f"Active DOFs exceed total DOFs: {invalid}")
            
        if np.any(self.active_dofs < 0):
            invalid = self.active_dofs[self.active_dofs < 0]
            errors.append(f"Negative active DOFs: {invalid}")
            
        if errors:
            error_msg = "Input validation failed:\n  " + "\n  ".join(errors)
            self.logger.critical(error_msg)
            raise ValueError(error_msg)

    def reconstruct(self) -> np.ndarray:
        """Execute full reconstruction pipeline with diagnostics."""
        start_time = time.perf_counter()
        self.logger.info("🚀 Starting displacement reconstruction")
        
        try:
            self._perform_mapping()
            self._validate_reconstruction()
            self._log_statistics()
            self._save_results()
            self._plot_displacements()
        except Exception as e:
            self.logger.critical(f"❌ Reconstruction failed: {str(e)}", exc_info=True)
            raise RuntimeError("Displacement reconstruction failed") from e
            
        self.reconstruction_time = time.perf_counter() - start_time
        self.logger.info(f"✅ Reconstruction completed in {self.reconstruction_time:.2f}s")
        return self.U_global

    def _perform_mapping(self):
        """Vectorized mapping of condensed displacements to global system."""
        # Use advanced indexing for O(1) mapping
        self.U_global[self.active_dofs] = self.U_cond
        
        # Validate fixed DOFs remain zero
        if self.fixed_dofs.size > 0:
            fixed_nonzero = np.nonzero(self.U_global[self.fixed_dofs])[0]
            if fixed_nonzero.size > 0:
                self.logger.warning(
                    f"Non-zero displacements at fixed DOFs: "
                    f"{self.fixed_dofs[fixed_nonzero]}"
                )

    def _validate_reconstruction(self):
        """Quality checks on reconstructed solution."""
        # Check NaN values
        nan_count = np.isnan(self.U_global).sum()
        if nan_count > 0:
            raise ValueError(f"{nan_count} NaN values in reconstructed displacements")
            
        # Check energy preservation
        active_energy = np.dot(self.U_cond, self.U_cond)
        global_energy = np.dot(self.U_global, self.U_global)
        energy_diff = abs(active_energy - global_energy)
        
        if energy_diff > 1e-12 * max(active_energy, global_energy):
            self.logger.warning(
                f"Energy discrepancy detected: {energy_diff:.2e} "
                f"(Global: {global_energy:.2e}, Active: {active_energy:.2e})"
            )

    def _log_statistics(self):
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

    def _save_results(self):
        """Save reconstructed displacements in multiple formats."""
        # Save as NumPy binary
        np.save(self.job_results_dir / "U_global.npy", self.U_global)
        
        # Save as CSV with metadata
        metadata = f"""# Reconstruction Metadata
# Active DOFs: {len(self.active_dofs)}
# Fixed DOFs: {len(self.fixed_dofs)}
# Total DOFs: {self.total_dofs}
# Timestamp: {datetime.now().isoformat()}
"""
        np.savetxt(
            self.job_results_dir / "U_global.csv",
            self.U_global,
            header=metadata + "GlobalDisplacement",
            comments='',
            delimiter=','
        )
        self.logger.info("💾 Saved reconstructed displacements in multiple formats")

    def _plot_displacements(self):
        """Generate displacement visualization plots."""
        plt.figure(figsize=(12, 6))
        
        # Plot displacement magnitude distribution
        plt.subplot(1, 2, 1)
        plt.hist(np.abs(self.U_global), bins=50, log=True)
        plt.title("Displacement Magnitude Distribution")
        plt.xlabel("|Displacement|")
        plt.ylabel("Count (log scale)")
        
        # Plot sorted displacements
        plt.subplot(1, 2, 2)
        plt.plot(np.sort(self.U_global), 'r-')
        plt.title("Sorted Displacement Values")
        plt.xlabel("Rank")
        plt.ylabel("Displacement")
        
        plt.tight_layout()
        plt.savefig(self.job_results_dir / "displacement_plots.png")
        plt.close()
        self.logger.info("📈 Generated displacement visualization plots")

    @property
    def solution(self) -> np.ndarray:
        """Get reconstructed displacement vector with copy protection."""
        return self.U_global.copy()

    def get_displacement(self, dof: int) -> float:
        """Safe accessor for individual DOF displacements."""
        if not 0 <= dof < self.total_dofs:
            raise ValueError(f"Invalid DOF index: {dof}")
        return self.U_global[dof]
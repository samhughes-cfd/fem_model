# processing_OOP/static/operations/primary_results.py
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from dataclasses import dataclass
from typing import List
import logging

@dataclass
class PrimaryResultSet:
    """Container for raw FEM solution outputs"""
    K_global: sp.csr_matrix
    F_global: np.ndarray
    K_mod: sp.csr_matrix
    F_mod: np.ndarray
    U_global: np.ndarray
    R_global: np.ndarray
    element_dof_maps: List[np.ndarray]

class ComputePrimaryResults:
    """Computes and stores direct solver outputs"""
    
    def __init__(self, assembler, solver, mesh_dict):
        """
        Parameters:
            assembler: Global system assembler instance
            solver: Boundary-conditioned system solver
            mesh_dict: Dictionary with 'elements' list
        """
        self.assembler = assembler
        self.solver = solver
        self.mesh = mesh_dict
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = None

    def compute(self) -> PrimaryResultSet:
        """Execute primary results pipeline"""
        self._validate_systems()
        
        self.results = PrimaryResultSet(
            K_global=self.assembler.K_global,
            F_global=self.assembler.F_global,
            K_mod=self.solver.K_mod,
            F_mod=self.solver.F_mod,
            U_global=self.solver.U_global,
            R_global=self._compute_reactions(),
            element_dof_maps=self._get_element_dofs()
        )
        self.logger.info("Primary results computed")
        return self.results

    def _compute_reactions(self) -> np.ndarray:
        """Calculate reaction forces: K_global * U_global - F_global"""
        return self.assembler.K_global @ self.solver.U_global - self.assembler.F_global

    def _get_element_dofs(self) -> List[np.ndarray]:
        """Extract DOF maps from all elements"""
        return [elem.assemble_global_dof_indices() for elem in self.mesh["elements"]]

    def _validate_systems(self):
        """Verify matrix dimensions match"""
        if self.assembler.K_global.shape != self.solver.K_mod.shape:
            raise ValueError(
                f"Global matrix dim mismatch: {self.assembler.K_global.shape} vs "
                f"{self.solver.K_mod.shape}"
            )
        if len(self.solver.U_global) != self.assembler.K_global.shape[0]:
            raise ValueError(
                f"Solution dim mismatch: U_global {len(self.solver.U_global)} vs "
                f"K_global {self.assembler.K_global.shape[0]}"
            )

    def save(self, output_dir: Path):
        """Save results to disk in compressed format"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sparse matrices in CSR components
        sp.save_npz(output_dir / "K_global.npz", self.results.K_global)
        sp.save_npz(output_dir / "K_mod.npz", self.results.K_mod)
        
        # Save dense arrays
        np.savez_compressed(
            output_dir / "F_results.npz",
            F_global=self.results.F_global,
            F_mod=self.results.F_mod,
            U_global=self.results.U_global,
            R_global=self.results.R_global,
            element_dof_maps=self.results.element_dof_maps
        )
        
        self.logger.info(f"Saved primary results to {output_dir}")
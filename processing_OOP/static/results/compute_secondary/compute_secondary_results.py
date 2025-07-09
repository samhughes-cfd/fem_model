# processing_OOP\static\results\compute_secondary_results.py

import numpy as np
from scipy.special import roots_legendre
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
import logging

@dataclass
class GaussPointData:
    xi: float
    x: float
    stress: np.ndarray  # 3x3 tensor
    strain: np.ndarray  # 3x3 tensor
    shear: float
    moment: float

@dataclass
class SecondaryResultSet:
    gauss_data: Dict[int, List[GaussPointData]] = field(default_factory=dict)
    nodal_energy: np.ndarray = field(default_factory=lambda: np.array([]))

class ComputeSecondaryResults:
    """Computes derived quantities at optimal evaluation points"""
    
    def __init__(self, primary_results, mesh_dict):
        """
        Parameters:
            primary_results: PrimaryResultSet instance
            mesh_dict: Dictionary with:
                - 'elements': List of element objects
                - 'node_coordinates': Node positions array
                - 'connectivity': Element-node connectivity
        """
        self.primary = primary_results
        self.mesh = mesh_dict
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = SecondaryResultSet()
        self.gauss_xi, _ = roots_legendre(3)  # 3-point quadrature

    def compute(self) -> SecondaryResultSet:
        """Execute secondary results pipeline"""
        self._validate_inputs()
        self._compute_gauss_quantities()
        self._project_nodal_energy()
        self.logger.info("Secondary results computed")
        return self.results

    def _validate_inputs(self):
        """Verify mesh contains required data"""
        required_keys = {'elements', 'node_coordinates', 'connectivity'}
        missing = required_keys - self.mesh.keys()
        if missing:
            raise ValueError(f"Missing mesh keys: {missing}")

    def _compute_gauss_quantities(self):
        """Compute element-level quantities at Gauss points"""
        for elem in self.mesh["elements"]:
            elem_id = elem.element_id
            dofs = elem.assemble_global_dof_indices()
            U_e = self.primary.U_global[dofs]
            
            self.results.gauss_data[elem_id] = []
            for xi in self.gauss_xi:
                # Transform to physical coordinates
                x = self._xi_to_x(elem, xi)
                
                # Compute mechanical quantities
                B = elem.get_b_matrix(xi)
                strain = B @ U_e
                stress = elem.constitutive_matrix @ strain
                shear, moment = elem.compute_internal_forces(xi, U_e)
                
                self.results.gauss_data[elem_id].append(
                    GaussPointData(
                        xi=xi,
                        x=x,
                        stress=stress,
                        strain=strain,
                        shear=shear,
                        moment=moment
                    )
                )

    def _xi_to_x(self, elem, xi: float) -> float:
        """Convert natural coordinate to physical position"""
        return (xi + 1) * (elem.L / 2) + elem.x_start

    def _project_nodal_energy(self):
        """Distribute strain energy to nodes using shape functions"""
        self.results.nodal_energy = np.zeros(len(self.mesh["node_coordinates"]))
        
        for elem in self.mesh["elements"]:
            conn = self.mesh["connectivity"][elem.element_id]
            for gp in self.results.gauss_data[elem.element_id]:
                N = elem.shape_functions(gp.xi)
                energy = 0.5 * np.tensordot(gp.stress, gp.strain) * elem.jacobian()
                for i, node_id in enumerate(conn):
                    self.results.nodal_energy[node_id] += N[i] * energy

    def save(self, output_dir: Path):
        """Save results with Gauss/nodal separation"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Gauss point data
        gauss_path = output_dir / "gauss_data"
        gauss_path.mkdir()
        for elem_id, gp_list in self.results.gauss_data.items():
            data = {
                'xi': [gp.xi for gp in gp_list],
                'x': [gp.x for gp in gp_list],
                'stress': [gp.stress for gp in gp_list],
                'strain': [gp.strain for gp in gp_list],
                'shear': [gp.shear for gp in gp_list],
                'moment': [gp.moment for gp in gp_list]
            }
            np.savez(gauss_path / f"element_{elem_id}.npz", **data)
        
        # Save nodal energy
        np.save(output_dir / "nodal_energy.npy", self.results.nodal_energy)
        
        self.logger.info(f"Saved secondary results to {output_dir}")
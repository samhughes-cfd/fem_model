# processing_OOP\static\results\compute_secondary\gaussian_to_nodal.py

import numpy as np
from typing import List, Dict
#from ...containers.nodal_results import NodalResults
#from ...containers.gaussian_results import GaussianResults
#from ...shape_function_library import ShapeFunctionRegistry  # assumed

# NODAL RESOLUTION

class GaussianToNodalProjector:
    """
    Projects Gaussian-level quantities (strain, stress, strain energy, section forces)
    to nodal resolution using shape function interpolation per element type.

    Each element's shape function operator is retrieved using its type from the element dictionary.
    """

    def __init__(
        self,
        element_dictionary: Dict[int, dict],
        grid_dictionary: Dict[int, dict],
        gaussian_results: GaussianResults,
        shape_function_registry: ShapeFunctionRegistry  # your central access point
    ):
        self.element_dictionary = element_dictionary
        self.grid_dictionary = grid_dictionary
        self.gaussian_results = gaussian_results
        self.shape_function_registry = shape_function_registry

        self.n_nodes = len(grid_dictionary)
        self.nodal_strain = np.zeros((self.n_nodes, 6))
        self.nodal_stress = np.zeros((self.n_nodes, 6))
        self.nodal_energy = np.zeros((self.n_nodes,))
        self.nodal_section_force = np.zeros((self.n_nodes, 6))
        self.weight = np.zeros((self.n_nodes,))

    def run(self) -> NodalResults:
        """Interpolates and returns NodalResults based on GaussianResults."""
        for e_id, element_data in self.element_dictionary.items():
            node_ids = element_data["node_ids"]
            element_type = element_data["type"]
            xi_gauss, _ = element_data["integration_points"]

            shape_fn_operator = self.shape_function_registry[element_type]

            for g, xi in enumerate(xi_gauss):
                N_vals = shape_fn_operator.natural_coordinate_form(xi)[0][0]

                ε = self.gaussian_results.strain[e_id][g]
                σ = self.gaussian_results.stress[e_id][g]
                w = self.gaussian_results.internal_energy_density[e_id][g]
                s = self.gaussian_results.internal_forces[e_id][g]

                for a, global_node_id in enumerate(node_ids):
                    N = N_vals[a]
                    self.nodal_strain[global_node_id] += N * ε
                    self.nodal_stress[global_node_id] += N * σ
                    self.nodal_energy[global_node_id] += N * w
                    self.nodal_section_force[global_node_id] += N * s
                    self.weight[global_node_id] += N

        # Normalize
        nonzero = self.weight > 0
        self.nodal_strain[nonzero] /= self.weight[nonzero, np.newaxis]
        self.nodal_stress[nonzero] /= self.weight[nonzero, np.newaxis]
        self.nodal_energy[nonzero] /= self.weight[nonzero]
        self.nodal_section_force[nonzero] /= self.weight[nonzero, np.newaxis]

        return NodalResults(
            strain=self.nodal_strain,
            stress=self.nodal_stress,
            strain_energy_density=self.nodal_energy,
            internal_forces=self.nodal_section_force
        )
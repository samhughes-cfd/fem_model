# processing_OOP\static\results\compute_secondary\strain_energy.py

# GAUSSIAN RESOLUTION

import numpy as np
from typing import List

class ComputeStrainEnergyDensity:
    """
    Computes strain energy density at each Gauss point.

    The strain energy per unit volume at a Gauss point is computed as:

        w = 0.5 * εᵀ * D * ε
          = 0.5 * εᵀ * σ

    where:
        - ε is the strain vector at a Gauss point
        - D is the constitutive matrix
        - σ is the stress vector (can be precomputed or derived via D @ ε)
    """

    def __init__(self, element, strains: List[np.ndarray], stresses: List[np.ndarray] = None):
        self.element = element
        self.strains = strains
        self.stresses = stresses  # Optional — if not given, compute via D @ ε
        self.D = element.section_operator.constitutive_matrix
        self.logger = element.logger_operator

    def run(self) -> List[np.ndarray]:
        """Compute strain energy density per Gauss point.

        Returns
        -------
        List[np.ndarray]
            Scalar strain energy density values (shape: [1,] or [()])
        """
        energy_list = []

        if self.logger:
            self.logger.log_text("strain_energy", f"\n=== Element {self.element.element_id} Strain Energy Computation ===")

        for i, ε in enumerate(self.strains):
            σ = self.stresses[i] if self.stresses else self.D @ ε
            energy = 0.5 * np.dot(ε, σ)  # scalar
            energy_list.append(np.array(energy))

            if self.logger:
                self.logger.log_scalar("strain_energy", energy, {
                    "name": f"Energy Density at Gauss Point #{i}"
                })

        if self.logger:
            self.logger.flush("strain_energy")

        return energy_list
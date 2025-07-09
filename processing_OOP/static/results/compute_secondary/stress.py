# processing_OOP\static\results\compute_secondary\stress_tensor.py

# GAUSSIAN RESOLUTION

import numpy as np
from typing import List

class ComputeStressTensor:
    """
    Computes stress tensors σ at Gauss points within an element.

    Stress is obtained from strain via the constitutive relation:

        σ = D @ ε

    where D is the section constitutive (material stiffness) matrix.
    """

    def __init__(self, element, strains: List[np.ndarray]):
        self.element = element
        self.strains = strains
        self.D = element.section_operator.constitutive_matrix
        self.logger = element.logger_operator

    def run(self) -> List[np.ndarray]:
        """Evaluate stress vectors at each Gauss point.

        Returns
        -------
        List[np.ndarray]
            List of stress vectors σ at each Gauss point (shape: [6,])
        """
        stresses = []

        if self.logger:
            self.logger.log_text("stress", f"\n=== Element {self.element.element_id} Stress Computation ===")

        for i, strain in enumerate(self.strains):
            stress = self.D @ strain  # σ = D * ε
            stresses.append(stress)

            if self.logger:
                self.logger.log_vector("stress", stress, {
                    "name": f"Stress at Gauss Point #{i}"
                })

        if self.logger:
            self.logger.flush("stress")

        return stresses
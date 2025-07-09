# processing_OOP\static\results\compute_secondary\strain_tensor.py

# GAUSSIAN RESOLUTION

import numpy as np
from typing import List

class ComputeStrainTensor:
    """
    Computes strain tensors ε at Gauss points within an element.

    These are derived from the element nodal displacement vector U_e and the
    strain-displacement matrix B at each integration point:

        ε = B(xi) @ U_e

    The resulting strain vectors may represent engineering strain components
    [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz] depending on element type.
    """

    def __init__(self, element):
        self.element = element
        self.shape_fn = element.shape_function_operator
        self.U_e = element.U_e
        self.xi_gauss, _ = element.integration_points
        self.logger = element.logger_operator

    def run(self) -> List[np.ndarray]:
        """Evaluate strain vectors at each Gauss point.

        Returns
        -------
        List[np.ndarray]
            List of strain vectors ε at each Gauss point (shape: [6,])
        """
        strains = []

        if self.logger:
            self.logger.log_text("strain", f"\n=== Element {self.element.element_id} Strain Computation ===")

        for xi in self.xi_gauss:
            _, B = self.shape_fn.natural_coordinate_form(xi)
            B = B[0]  # shape: (6, 12)

            strain = B @ self.U_e  # ε = B * u
            strains.append(strain)

            if self.logger:
                self.logger.log_vector("strain", strain, {
                    "name": f"Strain at xi = {xi:.3f}"
                })

        if self.logger:
            self.logger.flush("strain")

        return strains
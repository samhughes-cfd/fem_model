# processing_OOP\static\results\compute_secondary\section_force.py

# GAUSSIAN RESOLUTION

import numpy as np
from typing import List

class ComputeSectionForce:
    """
    Computes internal **section force resultants** [N, Vy, Vz, T, My, Mz] at Gauss points.

    These are the **physical stress-resultant vectors** at specific points along the element
    (usually Gauss integration points), and describe the internal state of the element due to
    deformation. They are used for **stress recovery**, **failure checks**, and **engineering interpretation**.

    Unlike the internal nodal force vector (K_e @ U_e), which expresses nodal reactions,
    section forces reflect local **internal equilibrium** across the cross-section. They are derived as:

        ε = B @ U_e                → strain at Gauss point
        σ = D @ ε                 → stress from material law
        [N, Vy, Vz, T, My, Mz]    → integrated from σ over the cross-section

    For solid or 2D models:
        N   = ∫_A σ_xx dA
        Vy  = ∫_A τ_xy dA
        Mz  = ∫_A y σ_xx dA
        (and so on)

    In 1D beam theory, these resultants are often directly obtained via interpolation of strain
    and stress at Gauss points.

    These section forces describe what the element "feels" internally at different cross-sections
    and are essential for interpreting internal loading conditions beyond nodal degrees of freedom.
    """

    def __init__(self, element):
        self.element = element
        self.shape_fn = element.shape_function_operator
        self.section = element.section_operator
        self.U_e = element.U_e
        self.xi_gauss, _ = element.integration_points
        self.logger = element.logger_operator

    def run(self) -> List[np.ndarray]:
        """Compute section force resultants at each Gauss point.

        Returns
        -------
        List[np.ndarray]
            List of section force vectors [N, Vy, Vz, T, My, Mz] for each Gauss point
        """
        section_forces = []

        if self.logger:
            self.logger.log_text("section_forces", f"\n=== Element {self.element.element_id} Section Force Computation ===")

        for xi in self.xi_gauss:
            # Get strain-displacement matrix B at Gauss point xi
            _, B = self.shape_fn.natural_coordinate_form(xi)
            B = B[0]  # shape: (6, 12)

            # Compute strain and stress
            strain = B @ self.U_e
            stress = self.section.constitutive_matrix @ strain

            # Map stress vector to section force resultants
            section_force = self._reduce_stress_to_section_forces(stress)
            section_forces.append(section_force)

            if self.logger:
                self.logger.log_vector("section_forces", section_force, {
                    "name": f"Section Forces at xi = {xi:.3f}"
                })

        if self.logger:
            self.logger.flush("section_forces")

        return section_forces

    def _reduce_stress_to_section_forces(self, stress: np.ndarray) -> np.ndarray:
        """Project the stress vector to section force resultants.

        Parameters
        ----------
        stress : np.ndarray
            Stress vector [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz]

        Returns
        -------
        np.ndarray
            Section force vector: [N, Vy, Vz, T, My, Mz]
        """
        # In typical beam theory, this stress vector is already reduced to sectional form
        return stress

# processing_OOP\static\results\compute_secondary\internal_force.py

# NODAL RESOLUTION

import numpy as np

class ComputeInternalForce:
    """
    Computes the internal (nodal) force vector F_int for a finite element.

    This vector is often referred to as the **element nodal internal force vector** because
    it expresses the internal forces the element "pushes back with" at its nodes in response
    to deformation. It is computed via:

        F_int = K_e @ U_e

    where:
        - K_e is the element stiffness matrix
        - U_e is the element nodal displacement vector

    These internal nodal forces are **virtual reaction forces** that, when assembled globally,
    enforce global equilibrium:

        ∑_e (K_e @ U_e) = F_ext

    While the vector has a nodal structure, its values represent **stress-resultants** derived
    from the strain-displacement matrix and constitutive behavior. They capture how the material
    and geometry resist external loads in terms of nodal degrees of freedom.
    """

    def __init__(self, element):
        self.element = element
        self.K_e = element.K_e
        self.U_e = element.U_e
        self.logger = element.logger_operator

    def run(self) -> np.ndarray:
        """Compute internal force vector: F_int = K_e @ U_e

        Returns
        -------
        np.ndarray
            Internal resisting force vector in local coordinates (shape: [12,])
        """
        if self.logger:
            self.logger.log_text("internal_force", f"\n=== Element {self.element.element_id} Internal Force Vector Computation ===")

        F_int = self.K_e @ self.U_e

        if self.logger:
            self.logger.log_matrix("internal_force", F_int.reshape(1, -1), {"name": "Internal Force Vector"})
            self.logger.flush("internal_force")

        return F_int
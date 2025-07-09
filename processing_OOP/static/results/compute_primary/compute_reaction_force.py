# processing_OOP\static\results\compute_primary\compute_reaction_force.py

import numpy as np

class ComputeReactionForce:
    def __init__(self, K_global, F_global, U_global, fixed_dofs):
        self.K_global = K_global
        self.F_global = F_global.reshape(-1)
        self.U_global = U_global.reshape(-1)
        self.fixed_dofs = fixed_dofs

        if np.any(self.fixed_dofs >= len(self.U_global)):
            raise IndexError("fixed_dofs contain indices out of range for U_global.")

    def compute(self) -> np.ndarray:
        """
        Computes the global reaction force vector.
        Only the fixed degrees of freedom will contain non-zero values.

        Returns:
            np.ndarray: Global reaction force vector with values only at fixed DOFs.
        """
        R_raw = self.K_global @ self.U_global - self.F_global
        R_global = np.zeros_like(R_raw)
        R_global[self.fixed_dofs] = R_raw[self.fixed_dofs]
        return R_global

# processing_OOP\static\results\compute_primary\compute_residual.py

import numpy as np

class ComputeResidual:
    def __init__(self, K_global, F_global, U_global):
        self.K_global = K_global
        self.F_global = F_global.reshape(-1)
        self.U_global = U_global.reshape(-1)

        if (K_global.shape[0] != self.F_global.shape[0] or
            K_global.shape[1] != self.U_global.shape[0]):
            raise ValueError("Incompatible dimensions for residual computation.")

    def compute(self) -> np.ndarray:
        """
        Compute the residual force vector:
        residual = F_global - K_global @ U_global

        Returns:
            np.ndarray: Residual vector indicating imbalance at each DOF.
        """
        return self.F_global - self.K_global @ self.U_global

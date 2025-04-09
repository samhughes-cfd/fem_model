# pre_processing\element_library\euler_bernoulli\utilities\B_matrix_6DOF.py

import numpy as np

def B_matrix(dN_dxi: np.ndarray, d2N_dxi2: np.ndarray, L: float) -> np.ndarray:
        """
        Construct the strain-displacement B-matrix for each Gauss point for a 2-node 3D Euler-Bernoulli element.

        Args:
            dN_dxi (np.ndarray): First derivatives of shape functions, shape (g, 12, 6)
            d2N_dxi2 (np.ndarray): Second derivatives of shape functions, shape (g, 12, 6)

        Returns:
            np.ndarray: Strain-displacement matrices at each Gauss point, shape (g, 4, 12)
        """
        detJ = L / 2
        g = dN_dxi.shape[0]
        B_matrix = np.zeros((g, 4, 12))

        for i in range(g):
            B = np.zeros((4, 12))

            # Axial strain εₓ = duₓ/dx
            B[0, 0] = dN_dxi[i, 0, 0] 
            B[0, 6] = dN_dxi[i, 6, 0]

            # Bending about Z-axis κ_z (XY plane bending: u_y, θ_z)
            B[1, [1, 7, 5, 11]] = [
                d2N_dxi2[i, 1, 1],   # u_y node 1
                d2N_dxi2[i, 7, 1],   # u_y node 2
                d2N_dxi2[i, 5, 5],   # θ_z node 1
                d2N_dxi2[i, 11, 5]   # θ_z node 2
            ]

            # Bending about Y-axis κ_y (XZ plane bending: u_z, θ_y)
            B[2, [2, 8, 4, 10]] = [
                d2N_dxi2[i, 2, 2],   # u_z node 1
                d2N_dxi2[i, 8, 2],   # u_z node 2
                d2N_dxi2[i, 4, 4],   # θ_y node 1
                d2N_dxi2[i, 10, 4]   # θ_y node 2
            ]

            # Torsional strain γₓ = dθₓ/dx
            B[3, 3] = dN_dxi[i, 3, 3]
            B[3, 9] = dN_dxi[i, 9, 3]

            B_matrix[i] = B

        return B_matrix
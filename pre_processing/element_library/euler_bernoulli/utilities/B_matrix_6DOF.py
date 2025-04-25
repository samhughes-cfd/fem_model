# pre_processing\element_library\euler_bernoulli\utilities\B_matrix_6DOF.py

import numpy as np

def B_matrix(dN_dxi: np.ndarray, d2N_dxi2: np.ndarray, L: float) -> np.ndarray:
    """
    Construct the strain-displacement B-matrix at each Gauss point for a 2-node 3D Euler-Bernoulli beam element.

    This function transforms derivatives of shape functions from the natural coordinate domain (ξ ∈ [-1, 1])
    into the physical coordinate domain (x ∈ [0, L]) using the standard isoparametric mapping:

        dN/dx  = (2 / L) · dN/dξ
        d²N/dx² = (4 / L²) · d²N/dξ²

    The resulting B-matrix expresses physical strain measures:
    - Axial strain ε_x
    - Bending curvatures κ_z and κ_y
    - Torsional strain γ_x

    All derivatives used in strain calculation are returned in global (Cartesian) coordinates.

    Args:
        dN_dxi (np.ndarray): First derivatives of shape functions w.r.t. ξ, shape (g, 12, 6)
        d2N_dxi2 (np.ndarray): Second derivatives of shape functions w.r.t. ξ, shape (g, 12, 6)
        L (float): Physical element length in global x-direction

    Returns:
        np.ndarray: Strain-displacement matrices, shape (g, 4, 12), for each Gauss point
    """

    detJ = L / 2
    g = dN_dxi.shape[0]
    B_matrix = np.zeros((g, 4, 12))

    for i in range(g):
        B = np.zeros((4, 12))

        # Axial strain ε_x = d(u_x)/dx
        B[0, 0] = dN_dxi[i, 0, 0] / detJ
        B[0, 6] = dN_dxi[i, 6, 0] / detJ

        # Bending about Z-axis κ_z (XY plane bending: u_y, θ_z)
        B[1, [1, 7, 5, 11]] = [
            d2N_dxi2[i, 1, 1],   # u_y node 1
            d2N_dxi2[i, 7, 1],   # u_y node 2
            d2N_dxi2[i, 5, 5],   # θ_z node 1
            d2N_dxi2[i, 11, 5]   # θ_z node 2
        ]  / detJ**2

        # Bending about Y-axis κ_y (XZ plane bending: u_z, θ_y)
        B[2, [2, 8, 4, 10]] = [
            d2N_dxi2[i, 2, 2],   # u_z node 1
            d2N_dxi2[i, 8, 2],   # u_z node 2
            d2N_dxi2[i, 4, 4],   # θ_y node 1
            d2N_dxi2[i, 10, 4]   # θ_y node 2
        ] / detJ**2

        # Torsional strain γ_x = d(θ_x)/dx
        B[3, 3] = dN_dxi[i, 3, 3] / detJ
        B[3, 9] = dN_dxi[i, 9, 3] / detJ

        B_matrix[i] = B

    return B_matrix
# pre_processing\element_library\euler_bernoulli\utilities\D_matrix_6DOF.py

import numpy as np

def D_matrix(E: float, G: float, A: float, I_y: float, I_z: float, I_x: float) -> np.ndarray:
    """
    Construct the material stiffness matrix D for a 3D Euler-Bernoulli beam element.

    Returns:
        D: (4x4) material stiffness matrix
    """
    D_matrix = np.diag([
        E * A,     # Axial
        E * I_z,   # Bending about Z
        E * I_y,   # Bending about Y
        G * I_x    # Torsion
    ])

    return D_matrix
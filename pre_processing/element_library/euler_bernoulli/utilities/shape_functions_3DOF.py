# pre_processing\element_library\utilities\shape_function_library\euler_bernoulli_sf.py

import numpy as np

def euler_bernoulli_shape_functions(xi, L, poly_order=3):
    """
    Computes the shape functions and their derivatives for an Euler-Bernoulli beam element.

    Parameters
    ----------
    xi : float or ndarray
        Natural coordinate(s) in [-1, 1]. Can be a scalar or a 1D array (n,).
    L : float
        Element length.
    poly_order : int, optional (default=3)
        Polynomial order (must be 3 for Euler-Bernoulli elements).

    Returns
    -------
    tuple (N_matrix, dN_dxi_matrix, d2N_dxi2_matrix)
        - **N_matrix** (ndarray, shape (n, 3, 6)): Shape function matrix.
        - **dN_dxi_matrix** (ndarray, shape (n, 3, 6)): First derivative w.r.t. ξ.
        - **d2N_dxi2_matrix** (ndarray, shape (n, 3, 6)): Second derivative w.r.t. ξ.
    """
    
    if poly_order != 3:
        raise ValueError("Euler-Bernoulli elements require cubic (3rd order) shape functions.")
    
    xi = np.atleast_1d(xi)  # Ensure xi is an array, shape: (n,)
    n = xi.shape[0]  # Number of evaluation points
    detJ = L / 2  # Jacobian determinant for 1D elements

    # --- AXIAL SHAPE FUNCTIONS (Linear Lagrange Interpolation) ---
    N1 = 0.5 * (1 - xi)  # Node 1
    N4 = 0.5 * (1 + xi)  # Node 2

    # First derivatives (constant for axial displacement)
    dN1_dxi = -0.5 * np.ones_like(xi)
    dN4_dxi =  0.5 * np.ones_like(xi)

    # Second derivatives (zero for linear functions)
    d2N1_dxi2 = np.zeros_like(xi)
    d2N4_dxi2 = np.zeros_like(xi)

    # --- BENDING SHAPE FUNCTIONS (Cubic Hermite Interpolation) ---
    N2 = 0.25 * (1 - xi)**2 * (1 + 2 * xi)  # Node 1
    N5 = 0.25 * (1 + xi)**2 * (1 - 2 * xi)  # Node 2

    # First derivatives
    dN2_dxi = 0.5 * (1 - xi) * (1 + 2 * xi) - 0.25 * (1 - xi)**2 * 2
    dN5_dxi = -0.5 * (1 + xi) * (1 - 2 * xi) + 0.25 * (1 + xi)**2 * (-2)

    # Second derivatives
    d2N2_dxi2 = 1.5 * xi - 0.5
    d2N5_dxi2 = -1.5 * xi + 0.5

    # --- ROTATIONAL SHAPE FUNCTIONS (Derived from Bending) ---
    N3 = (1 / detJ) * dN2_dxi  # Node 1 rotation
    N6 = (1 / detJ) * dN5_dxi  # Node 2 rotation

    # First derivatives (Curvature shape functions)
    dN3_dxi = (1 / detJ) * d2N2_dxi2
    dN6_dxi = (1 / detJ) * d2N5_dxi2

    # Second derivatives (Higher order curvature terms, rarely used)
    d2N3_dxi2 = (1 / detJ) * (1.5 * np.ones_like(xi))
    d2N6_dxi2 = (1 / detJ) * (-1.5 * np.ones_like(xi))

    # --- Assemble the Shape Function Matrices (3 × 6) ---
    # Axial DOFs (row 0), Transverse DOFs (row 1), Rotational DOFs (row 2)

    N_matrix = np.stack((
        np.column_stack((N1, np.zeros(n), np.zeros(n), N4, np.zeros(n), np.zeros(n))),  # Axial row
        np.column_stack((np.zeros(n), N2, np.zeros(n), np.zeros(n), N5, np.zeros(n))),  # Transverse displacement row
        np.column_stack((np.zeros(n), np.zeros(n), N3 , np.zeros(n), np.zeros(n), N6))   # Rotation row (N3, N6 derived)
    ), axis=1)  # Shape: (n, 3, 6)

    dN_dxi_matrix = np.stack((
        np.column_stack((dN1_dxi, np.zeros(n), np.zeros(n), dN4_dxi, np.zeros(n), np.zeros(n))),  # Axial row
        np.column_stack((np.zeros(n), dN2_dxi, np.zeros(n), np.zeros(n), dN5_dxi, np.zeros(n))),  # Transverse row
        np.column_stack((np.zeros(n), np.zeros(n), dN3_dxi, np.zeros(n), np.zeros(n), dN6_dxi))   # Rotation row (first derivative)
    ), axis=1)  # Shape: (n, 3, 6)

    d2N_dxi2_matrix = np.stack((
        np.column_stack((d2N1_dxi2, np.zeros(n), np.zeros(n), d2N4_dxi2, np.zeros(n), np.zeros(n))),  # Axial row
        np.column_stack((np.zeros(n), d2N2_dxi2, np.zeros(n), np.zeros(n), d2N5_dxi2, np.zeros(n))),  # Transverse row
        np.column_stack((np.zeros(n), np.zeros(n), d2N3_dxi2, np.zeros(n), np.zeros(n), d2N6_dxi2))   # Curvature row
    ), axis=1)  # Shape: (n, 3, 6)

    return N_matrix, dN_dxi_matrix, d2N_dxi2_matrix
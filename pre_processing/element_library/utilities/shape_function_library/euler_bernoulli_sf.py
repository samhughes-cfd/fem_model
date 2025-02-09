# pre_processing\element_library\utilities\shape_function_library\euler_bernoulli_sf.py

import numpy as np

def euler_bernoulli_shape_functions(xi, L, poly_order=3):
    """
    Compute the shape functions for an Euler-Bernoulli beam element.

    Parameters:
        xi (float): Natural coordinate in [-1, 1].
        L (float): Element length
        poly_order (int): Polynomial order of the shape functions (default = 3 for Euler-Bernoulli)

    Returns:
        tuple: (N, dN_dxi, d2N_dxi2)
            N (ndarray): Shape function vector (6,)
            dN_dxi (ndarray): First derivatives of shape functions w.r.t. xi (6,)
            d2N_dxi2 (ndarray): Second derivatives of shape functions w.r.t. xi (6,)

    Notes:
        - Rotation shape functions (N3, N6) are derived as the first derivative of transverse displacement shape functions (N2, N5).
        - Since Euler-Bernoulli beam theory assumes no shear deformation, the slope (rotation) is given by:
          \theta(x) = dv/dx.
        - Given that x = (L/2) * xi, differentiation follows:
          d/dx = (2/L) * d/dxi.
        - Thus, rotation shape functions are obtained as:
          N3 = (2/L) * dN2/dxi, and N6 = (2/L) * dN5/dxi.
        - This transformation ensures proper coupling between transverse displacement and rotation within the element.
    """
    if poly_order != 3:
        raise ValueError("Euler-Bernoulli elements require cubic (3rd order) shape functions.")

    # Axial Shape Functions (Linear Lagrange)
    N1 = 0.5 * (1 - xi)  # Axial displacement at Node 1
    N4 = 0.5 * (1 + xi)  # Axial displacement at Node 2

    # Transverse Displacement Shape Functions (Cubic Hermite)
    N2 = 0.25 * (1 - xi)**2 * (1 + 2*xi)  # Transverse displacement at Node 1
    N5 = 0.25 * (1 + xi)**2 * (1 - 2*xi)  # Transverse displacement at Node 2

    # Rotation Shape Functions (First derivative of transverse displacement)
    N3 = (L/8) * (1 - xi)**2 * (1 + xi)  # Rotation at Node 1
    N6 = (L/8) * (1 + xi)**2 * (1 - xi)  # Rotation at Node 2

    # First Derivatives, dN/dxi
    dN1_dxi = -0.5
    dN4_dxi = 0.5

    dN2_dxi = 0.5 * (1 - xi) * (1 + 2*xi) - 0.25 * (1 - xi)**2 * 2
    dN5_dxi = -0.5 * (1 + xi) * (1 - 2*xi) + 0.25 * (1 + xi)**2 * (-2)

    dN3_dxi = (L/8) * (3*xi**2 - 1 - 2*xi)
    dN6_dxi = (L/8) * (3*xi**2 - 1 + 2*xi)

    # Second Derivatives, d2N/dxi2
    d2N1_dxi2 = 0.0
    d2N4_dxi2 = 0.0

    d2N2_dxi2 = 1.5 * xi - 0.5
    d2N5_dxi2 = -1.5 * xi + 0.5

    d2N3_dxi2 = (3*L/8) * (2*xi - 1)
    d2N6_dxi2 = (3*L/8) * (2*xi + 1)

    # Assemble Vectors N, dN_dxi, d2N_dxi2
    N = np.array([N1, N2, N3, N4, N5, N6])
    dN_dxi = np.array([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi])
    d2N_dxi2 = np.array([d2N1_dxi2, d2N2_dxi2, d2N3_dxi2, d2N4_dxi2, d2N5_dxi2, d2N6_dxi2])

    return N, dN_dxi, d2N_dxi2
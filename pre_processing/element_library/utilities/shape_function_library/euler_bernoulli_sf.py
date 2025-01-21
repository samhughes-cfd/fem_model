# pre_processing\element_library\utilities\shape_function_library\euler_bernoulli_sf.py

import numpy as np

def euler_bernoulli_shape_functions(xi, L, poly_order=3):
    """
    Compute the shape functions for an Euler-Bernoulli beam element.

    ### Euler-Bernoulli Beam Theory:
        - Assumes plane sections remain normal to the N.A. (no shear deformation).
        - Rotation is derived from the slope of transverse displacement: theta_z = du_y/dx
        - Cubic Hermite polynomials are required for transverse displacement to ensure that its derivative (rotation) follows a quadratic shape function.
        - Well posed for slender beams where shear deformation is negligible.

    ### Degrees of Freedom (DOFs) per Node:
        Node 1:
        - N1 (index 0): Axial displacement
        - N2 (index 1): Transverse displacement
        - N3 (index 2): Rotation due to bending

        Node 2:
        - N4 (index 3): Axial displacement
        - N5 (index 4): Transverse displacement
        - N6 (index 5): Rotation due to bending

    Parameters:
        xi (float): Natural coordinate in [-1, 1].
        L (float): Element length
        poly_order (int): Polynomial order of the shape functions (default = 3 for Euler-Bernoulli)

    Returns:
        tuple: (N, dN_dxi, d2N_dxi2)
            N (ndarray): Shape function vector (6,)
            dN_dxi (ndarray): First derivatives of shape functions w.r.t. xi (6,)
            d2N_dxi2 (ndarray): Second derivatives of shape functions w.r.t. xi (6,)
    """

    if poly_order != 3:
        raise ValueError("Euler-Bernoulli elements require cubic (3rd order) shape functions.")

    # Axial Shape Functions (Linear)
    N1 = 0.5 * (1 - xi)  # Axial displacement at Node 1
    N4 = 0.5 * (1 + xi)  # Axial displacement at Node 2

    # Transverse Displacement Shape Functions (Cubic Hermite)
    N2 = 0.25 * (1 - xi) ** 2 * (2 + xi)  # Transverse displacement at Node 1
    N5 = 0.25 * (1 + xi) ** 2 * (2 - xi)  # Transverse displacement at Node 2

    # Rotation Shape Functions (Derived from du_y/dx, Quadratic)
    N3 = 0.125 * L * (1 - xi) ** 2 * (1 + xi)  # Rotation at Node 1
    N6 = 0.125 * L * (1 + xi) ** 2 * (xi - 1)  # Rotation at Node 2

    # First Derivatives, dN/dxi
    dN1_dxi = -0.5
    dN4_dxi = 0.5

    dN2_dxi = 0.5 * (1 - xi) * (2 + xi) - 0.25 * (1 - xi) ** 2
    dN5_dxi = -0.5 * (1 + xi) * (2 - xi) + 0.25 * (1 + xi) ** 2

    dN3_dxi = 0.125 * L * (3 * xi**2 - 1 - 2 * xi)
    dN6_dxi = 0.125 * L * (3 * xi**2 - 1 + 2 * xi)

    # Second Derivatives, d2N/dxi2
    d2N1_dxi2 = 0.0
    d2N4_dxi2 = 0.0

    d2N2_dxi2 = 1.5 * xi - 0.5
    d2N5_dxi2 = -1.5 * xi + 0.5

    d2N3_dxi2 = 0.375 * L * (6 * xi - 1)
    d2N6_dxi2 = 0.375 * L * (6 * xi + 1)

    # Assemble Vectors N, dN_dxi, d2N_dxi2
    N = np.array([N1, N2, N3, N4, N5, N6])
    dN_dxi = np.array([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi])
    d2N_dxi2 = np.array([d2N1_dxi2, d2N2_dxi2, d2N3_dxi2, d2N4_dxi2, d2N5_dxi2, d2N6_dxi2])

    return N, dN_dxi, d2N_dxi2
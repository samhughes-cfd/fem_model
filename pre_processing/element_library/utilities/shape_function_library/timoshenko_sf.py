# pre_processing/element_library/timoshenko/shape_functions_library.py

import numpy as np

def timoshenko_shape_functions(xi, L, poly_order=2):
    """
    Compute the shape functions for a Timoshenko beam element including shear deformation.

    ### Timoshenko Beam Theory:
        - Shear deformation is explicitly modeled (plane sections do not necessarily remain normal to N.A.).
        - Rotation theta_z is an independent degree of freedom, rather than being derived from displacement.
        - Quadratic shape functions for both transverse displacement and rotation.
        - Well posed for thicker beams and higher-frequency responses, where shear effects are significant.

    ### Degrees of Freedom (DOFs) per Node:
        Node 1:
        - N1 (index 0): Axial displacement
        - N2 (index 1): Transverse displacement
        - N3 (index 2): Rotation (bending + shear)

        Node 2:
        - N4 (index 3): Axial displacement
        - N5 (index 4): Transverse displacement
        - N6 (index 5): Rotation (bending + shear)

    Parameters:
        xi (float): Natural coordinate in [-1, 1].
        L (float): Element length
        poly_order (int): Polynomial order of the shape functions (default = 2 for Timoshenko)

    Returns:
        tuple: (N, dN_dxi, d2N_dxi2)
    """

    if poly_order != 2:
        raise ValueError("Timoshenko elements use quadratic (2nd order) shape functions.")

    # Axial Shape Functions (Linear, same as EB)
    N1 = 0.5 * (1 - xi)  # Axial displacement at Node 1
    N4 = 0.5 * (1 + xi)  # Axial displacement at Node 2

    # Transverse Shape Functions (Quadratic due to shear)
    N2 = 0.5 * (1 - xi)  # Transverse displacement at Node 1
    N5 = 0.5 * (1 + xi)  # Transverse displacement at Node 2

    # Rotation Shape Functions (Independent DOF, Quadratic)
    N3 = (L / 4) * (1 - xi**2)  # Rotation at Node 1
    N6 = (L / 4) * (xi**2 - 1)  # Rotation at Node 2

    # First Derivatives, dN/dxi
    dN1_dxi = -0.5  # Axial displacement at Node 1
    dN4_dxi = 0.5   # Axial displacement at Node 2

    dN2_dxi = -0.5  # Transverse displacement at Node 1
    dN5_dxi = 0.5   # Transverse displacement at Node 2

    dN3_dxi = -0.5 * L * xi  # Rotation at Node 1
    dN6_dxi = 0.5 * L * xi   # Rotation at Node 2

    # Second Derivatives, d2N/dxi2
    d2N1_dxi2 = 0.0  # Axial displacement at Node 1
    d2N4_dxi2 = 0.0  # Axial displacement at Node 2

    d2N2_dxi2 = 0.0  # Transverse displacement at Node 1
    d2N5_dxi2 = 0.0  # Transverse displacement at Node 2

    d2N3_dxi2 = -0.5 * L  # Rotation at Node 1
    d2N6_dxi2 = 0.5 * L   # Rotation at Node 2

    # **Assemble Vectors**
    N = np.array([N1, N2, N3,
                   N4, N5, N6])
    dN_dxi = np.array([dN1_dxi, dN2_dxi, dN3_dxi,
                        dN4_dxi, dN5_dxi, dN6_dxi])
    d2N_dxi2 = np.array([d2N1_dxi2, d2N2_dxi2, d2N3_dxi2,
                          d2N4_dxi2, d2N5_dxi2, d2N6_dxi2])

    return N, dN_dxi, d2N_dxi2
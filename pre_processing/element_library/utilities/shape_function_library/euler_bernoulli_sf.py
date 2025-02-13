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
      - **N_matrix** (ndarray, shape (n, 2, 6)): Shape function matrix.
      - **dN_dxi_matrix** (ndarray, shape (n, 2, 6)): First derivative w.r.t. ξ.
      - **d2N_dxi2_matrix** (ndarray, shape (n, 2, 6)): Second derivative w.r.t. ξ.

  Element System
  --------------
  A two-node Euler-Bernoulli beam element with **three degrees of freedom per node**:
    - **Axial displacement**: \( N_1, N_4 \) (Nodes 1, 2)
    - **Transverse displacement**: \( N_2, N_5 \) (Nodes 1, 2)
    - **Rotation (slope from transverse displacement)**: \( N_3, N_6 \) (Nodes 1, 2)

  Tensor Structure
  ----------------
  The returned matrices have shape **(n, 2, 6)**:
  - **Axis 0 (n)**: Evaluation points (layers).
  - **Axis 1 (2)**: Displacement type:
      - `0` → Axial displacements (\(N_1, N_4\))
      - `1` → Transverse & rotational (\(N_2, N_3, N_5, N_6\))
  - **Axis 2 (6)**: Shape functions for element degrees of freedom.

  Example Indexing
  ----------------
  - `N_matrix[i, :, :]` → (2,6) shape function matrix at Gauss point `i`.
  - `N_matrix[:, 0, :]` → Axial shape functions across all Gauss points.
  - `N_matrix[:, 1, :]` → Transverse & rotational shape functions across all Gauss points.

  Theoretical Notes
  -----------------
  - The rotation shape functions (N3, N6) are derived as the first derivative of the transverse displacement shape functions (N2, N5).
  - Euler-Bernoulli beam theory assumes no shear deformation, meaning the rotation is given by the derivative of the transverse displacement: θz = du_y/dx
  - Using the transformation from the natural coordinate ξ to the physical coordinate x, where:

    x = (L/2) * xi,
    
    We obtain the relation:

    d/dx = (2/L) d/dxi

  - Applying this transformation, the rotation shape functions are computed as:

    N3 = (2/L) (dN2/dξ),
    N6 = (2/L) (dN5/dξ)

    This transformation ensures that the transverse displacement and the rotation are properly coupled within the element.
  """


    if poly_order != 3:
        raise ValueError("Euler-Bernoulli elements require cubic (3rd order) shape functions.")
    
    # Ensure xi is a NumPy array.
    # If a scalar is passed, np.atleast_1d converts it to an array of shape (1,).
    xi = np.atleast_1d(xi)  # shape: (n,)
    n = xi.shape[0] # Number of evaluation points
    
    # (1) compute the scalar shape functions and their derivatives, each computed array is vectorized and has shape (n,)

    # --- Axial Shape Functions (Linear Lagrange) ---
    # N1: axial displacement at node 1, N4: axial displacement at node 2.
    N1 = 0.5 * (1 - xi)     # shape: (n,)
    N4 = 0.5 * (1 + xi)     # shape: (n,)
    
    # --- Transverse Displacement Shape Functions (Cubic Hermite) ---
    # N2: transverse displacement at node 1, N5: transverse displacement at node 2.
    N2 = 0.25 * (1 - xi)**2 * (1 + 2*xi)  # shape: (n,)
    N5 = 0.25 * (1 + xi)**2 * (1 - 2*xi)  # shape: (n,)
    
    # --- Rotation Shape Functions (Derived from transverse displacement) ---
    # N3: rotation at node 1, N6: rotation at node 2.
    N3 = (L/8) * (1 - xi)**2 * (1 + xi)   # shape: (n,)
    N6 = (L/8) * (1 + xi)**2 * (1 - xi)   # shape: (n,)
    
    # --- First Derivatives dN/dxi ---
    # Constant derivatives are expanded to arrays with the same shape as xi.
    dN1_dxi = -0.5 * np.ones_like(xi)  # shape: (n,)
    dN4_dxi =  0.5 * np.ones_like(xi)  # shape: (n,)
    
    dN2_dxi = 0.5 * (1 - xi) * (1 + 2*xi) - 0.25 * (1 - xi)**2 * 2  # shape: (n,)
    dN5_dxi = -0.5 * (1 + xi) * (1 - 2*xi) + 0.25 * (1 + xi)**2 * (-2)  # shape: (n,)
    
    dN3_dxi = (L/8) * (3*xi**2 - 1 - 2*xi)  # shape: (n,)
    dN6_dxi = (L/8) * (3*xi**2 - 1 + 2*xi)  # shape: (n,)
    
    # --- Second Derivatives d2N/dxi2 ---
    d2N1_dxi2 = np.zeros_like(xi)  # shape: (n,)
    d2N4_dxi2 = np.zeros_like(xi)  # shape: (n,)
    
    d2N2_dxi2 = 1.5 * xi - 0.5  # shape: (n,)
    d2N5_dxi2 = -1.5 * xi + 0.5  # shape: (n,)
    
    d2N3_dxi2 = (3*L/8) * (2*xi - 1)  # shape: (n,)
    d2N6_dxi2 = (3*L/8) * (2*xi + 1)  # shape: (n,)

    # (2) Assemble the shape function and deriavtive computed arrays into matrices for each evaluation point n

    # --- Assemble N_matrix ---
    N_matrix = np.stack((
      np.column_stack((N1, np.zeros(n), np.zeros(n), N4, np.zeros(n), np.zeros(n))),  # axial row: shape (n, 6)
      np.column_stack((np.zeros(n), N2, N3, np.zeros(n), N5, N6))  # transverse row: shape (n, 6)
    ), axis=1)  # for each Gauss point n we have (2,6), therefore shape: (n, 2, 6)

    # --- Assemble dN_dxi_matrix ---
    dN_dxi_matrix = np.stack((
      np.column_stack((dN1_dxi, np.zeros(n), np.zeros(n), dN4_dxi, np.zeros(n), np.zeros(n))),  # axial row: shape (n, 6)
      np.column_stack((np.zeros(n), dN2_dxi, dN3_dxi, np.zeros(n), dN5_dxi, dN6_dxi))  # transverse row: shape (n, 6)
    ), axis=1)  # for each Gauss point n we have (2,6), therefore shape: (n, 2, 6)

    # --- Second Derivatives d2N/dxi2 ---
    d2N_dxi2_matrix = np.stack((
      np.column_stack((d2N1_dxi2, np.zeros(n), np.zeros(n), d2N4_dxi2, np.zeros(n), np.zeros(n))),  # axial row: shape (n, 6)
      np.column_stack((np.zeros(n), d2N2_dxi2, d2N3_dxi2, np.zeros(n), d2N5_dxi2, d2N6_dxi2))  # transverse row: shape (n, 6)
    ), axis=1)  # for each Gauss point n we have (2,6), therefore shape: (n, 2, 6)

    # (3) Return the matrices

    return N_matrix, dN_dxi_matrix, d2N_dxi2_matrix
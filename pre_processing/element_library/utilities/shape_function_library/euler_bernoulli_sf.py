# pre_processing\element_library\utilities\shape_function_library\euler_bernoulli_sf.py

import numpy as np

def shape_functions(self, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Compute shape functions and derivatives for 3D Euler-Bernoulli beam.

  Args:
      xi: Natural coordinates in range [-1, 1]

  Returns:
      - N_matrix (g, 12, 6): Shape functions for translation and rotation DOFs
      - dN_dxi_matrix (g, 12, 6): First derivatives
      - d2N_dxi2_matrix (g, 12, 6): Second derivatives
  """
  xi = np.atleast_1d(xi)
  g = xi.shape[0]  # Number of Gauss points
  L = self.L  # Element length

  # **1. Axial Shape Functions (Linear Lagrange)**
  N1 = 0.5 * (1 - xi)
  N7 = 0.5 * (1 + xi)
  dN1_dxi = -0.5 * np.ones(g)
  dN7_dxi = 0.5 * np.ones(g)
  d2N1_dxi2 = np.zeros(g)
  d2N7_dxi2 = np.zeros(g)

  # **2. Bending in XY Plane (Hermite Cubic)**
  N2 = 0.5 * (1 - 3 * xi**2 + 2 * xi**3)  # Displacement
  N8 = 0.5 * (3 * xi**2 - 2 * xi**3 + 1)  # Displacement
  N3 = 0.5 * L * (xi - 2 * xi**2 + xi**3)  # Rotation θ
  N9 = 0.5 * L * (-xi**2 + xi**3)  # Rotation θ

  # **Derivatives**
  dN2_dxi = -3 * xi + 2 * xi**2
  dN8_dxi = 3 * xi - 2 * xi**2
  dN3_dxi = 0.5 * L * (1 - 4 * xi + 3 * xi**2)
  dN9_dxi = 0.5 * L * (-2 * xi + 3 * xi**2)

  d2N2_dxi2 = -3 + 4 * xi
  d2N8_dxi2 = 3 - 4 * xi
  d2N3_dxi2 = 0.5 * L * (-4 + 6 * xi)
  d2N9_dxi2 = 0.5 * L * (-2 + 6 * xi)

  # **3. Bending in XZ Plane (Reuse Hermite Cubic Functions)**
  N4, N10 = N2, N8
  N5, N11 = N3, N9
  dN4_dxi, dN10_dxi = dN2_dxi, dN8_dxi
  dN5_dxi, dN11_dxi = dN3_dxi, dN9_dxi
  d2N4_dxi2, d2N10_dxi2 = d2N2_dxi2, d2N8_dxi2
  d2N5_dxi2, d2N11_dxi2 = d2N3_dxi2, d2N9_dxi2

  # **4. Torsion Shape Functions (Linear Interpolation)**
  N6, N12 = N1, N7
  dN6_dxi, dN12_dxi = dN1_dxi, dN7_dxi
  d2N6_dxi2, d2N12_dxi2 = d2N1_dxi2, d2N7_dxi2

  # **5. Assemble Shape Function Matrices**
  N_matrix = np.zeros((g, 12, 6))
  dN_dxi_matrix = np.zeros((g, 12, 6))
  d2N_dxi2_matrix = np.zeros((g, 12, 6))

  ### **Axial DOF (u_x - along the beam length)**
  # Linear Lagrange shape functions for axial displacement
  N_matrix[:, 0, 0] = N1   # Node 1 axial displacement (u_x)
  N_matrix[:, 6, 0] = N7   # Node 2 axial displacement (u_x)

  ### **Transverse DOF (u_y - bending in the XZ plane)**
  # Hermite cubic shape functions for transverse displacement in Y-direction
  N_matrix[:, 1, 1] = N2   # Node 1 transverse displacement (u_y)
  N_matrix[:, 7, 1] = N8   # Node 2 transverse displacement (u_y)

  ### **Transverse DOF (u_z - bending in the XY plane)**
  # Hermite cubic shape functions for transverse displacement in Z-direction
  N_matrix[:, 2, 2] = N4   # Node 1 transverse displacement (u_z)
  N_matrix[:, 8, 2] = N10  # Node 2 transverse displacement (u_z)

  ### **Torsion DOF (θ_x - twist about the beam axis)**
  # Linear shape functions for torsion
  N_matrix[:, 3, 3] = N6   # Node 1 torsion (θ_x)
  N_matrix[:, 9, 3] = N12  # Node 2 torsion (θ_x)

  ### **Rotation DOF (θ_y - rotation about Y-axis, associated with u_z)**
  # Hermite cubic shape functions for bending rotation in the XY plane
  N_matrix[:, 4, 4] = N5   # Node 1 rotation (θ_y)
  N_matrix[:, 10, 4] = N11 # Node 2 rotation (θ_y)

  ### **Rotation DOF (θ_z - rotation about Z-axis, associated with u_y)**
  # Hermite cubic shape functions for bending rotation in the XZ plane
  N_matrix[:, 5, 5] = N3   # Node 1 rotation (θ_z)
  N_matrix[:, 11, 5] = N9  # Node 2 rotation (θ_z)

  # ---------------------------------------------------------------------------

  # **First Derivative Assignments (Strains & Curvatures)**

  ### **Axial Strain (ε_x = du_x/dx)**
  dN_dxi_matrix[:, 0, 0] = dN1_dxi  # Node 1 axial derivative
  dN_dxi_matrix[:, 6, 0] = dN7_dxi  # Node 2 axial derivative

  ### **Bending Curvature in XZ plane (κ_z = d²u_y/dx²)**
  dN_dxi_matrix[:, 1, 1] = dN2_dxi  # Node 1 bending derivative (u_y)
  dN_dxi_matrix[:, 7, 1] = dN8_dxi  # Node 2 bending derivative (u_y)

  ### **Bending Curvature in XY plane (κ_y = d²u_z/dx²)**
  dN_dxi_matrix[:, 2, 2] = dN4_dxi  # Node 1 bending derivative (u_z)
  dN_dxi_matrix[:, 8, 2] = dN10_dxi # Node 2 bending derivative (u_z)

  ### **Torsional Strain (γ_x = dθ_x/dx)**
  dN_dxi_matrix[:, 3, 3] = dN6_dxi  # Node 1 torsion derivative (θ_x)
  dN_dxi_matrix[:, 9, 3] = dN12_dxi # Node 2 torsion derivative (θ_x)

  ### **Rotational Derivatives (dθ_y/dx and dθ_z/dx)**
  dN_dxi_matrix[:, 4, 4] = dN5_dxi   # Node 1 bending rotation (θ_y)
  dN_dxi_matrix[:, 10, 4] = dN11_dxi # Node 2 bending rotation (θ_y)
  dN_dxi_matrix[:, 5, 5] = dN3_dxi   # Node 1 bending rotation (θ_z)
  dN_dxi_matrix[:, 11, 5] = dN9_dxi  # Node 2 bending rotation (θ_z)

  # ---------------------------------------------------------------------------

  # **Second Derivative Assignments (For Bending Moments Only)**
  # Second derivatives are only relevant for bending (curvatures).

  ### **Bending Curvature in XZ plane (κ_z = d²u_y/dx²)**
  d2N_dxi2_matrix[:, 1, 1] = d2N2_dxi2  # Node 1 curvature (u_y)
  d2N_dxi2_matrix[:, 7, 1] = d2N8_dxi2  # Node 2 curvature (u_y)

  ### **Bending Curvature in XY plane (κ_y = d²u_z/dx²)**
  d2N_dxi2_matrix[:, 2, 2] = d2N4_dxi2  # Node 1 curvature (u_z)
  d2N_dxi2_matrix[:, 8, 2] = d2N10_dxi2 # Node 2 curvature (u_z)

  ### **Rotational Second Derivatives (d²θ_y/dx² and d²θ_z/dx²)**
  d2N_dxi2_matrix[:, 5, 5] = d2N3_dxi2  # Node 1 rotation (θ_z)
  d2N_dxi2_matrix[:, 11, 5] = d2N9_dxi2 # Node 2 rotation (θ_z)
        
  return N_matrix, dN_dxi_matrix, d2N_dxi2_matrix
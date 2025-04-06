# pre_processing\element_library\euler_bernoulli\utilities\shape_functions_6DOF.py

import numpy as np
from typing import Tuple

def shape_functions(self, xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute shape functions and derivatives for 3D Euler-Bernoulli beam.

        ====================================================
        Node	Index   DOF      SF      Mode
        ====================================================
        1       0	    u_x      N1      Axial
        -       1	    u_y      N2      Bending (XY Plane)
        -       2	    u_z      N3      Bending (XZ Plane)
        -       3	    θ_x      N4      Torsional
        -       4	    θ_y      N5      Bending (XZ Plane)
        -       5	    θ_z      N6      Bending (XY Plane)
        2       6	    u_x      N7      Axial
        -       7	    u_y      N8      Bending (XY Plane)
        -       8	    u_z      N9      Bending (XZPlane)
        -       9	    θ_x      N10     Torsional
        -       10	    θ_y      N11     Bending (XZ Plane)
        -       11	    θ_z      N12     Bending (XY Plane)

        Args:
            xi: Natural coordinates in range [-1, 1]

        Returns:
            - N_matrix (g, 12, 6): Shape functions for translation and rotation DOFs
            - dN_dxi_matrix (g, 12, 6): First derivatives
            - d2N_dxi2_matrix (g, 12, 6): Second derivatives
        """
        xi = np.atleast_1d(xi)
        g = xi.shape[0]  # no. of gauss points
        L = self.L  # element length (x_end - x_start)

        # ====================================================================
        #                            SHAPE FUNCTIONS           
        # ====================================================================

        # ====================================================================
        # 1. Axial Shape Functions (Linear Lagrange Functions)
        # ====================================================================

        # Translation shape functions (u_x --> N1, N7)

        N1 = 0.5 * (1 - xi)                            # Node 1: u_x
        N7 = 0.5 * (1 + xi)                            # Node 2: u_x

        dN1_dxi = -0.5 * np.ones(g)                    # Node 1: d(u_x)/dx
        dN7_dxi = 0.5 * np.ones(g)                     # Node 2: d(u_x)/ dx
 
        d2N1_dxi2 = np.zeros(g)                        # Node 1: d2(u_x)/ dx2
        d2N7_dxi2 = np.zeros(g)                        # Node 2: d2(u_x)/ dx2

        # ====================================================================
        # 2. Bending in XY Plane (Hermite Cubic Functions)
        # ====================================================================

        # Translation shape functions (u_y --> N2, N8)

        N2  = 1 - 3 * xi**2 + 2 * xi**3                # Node 1: u_y
        N8  = 3 * xi**2 - 2 * xi**3                    # Node 2: u_y

        dN2_dxi  = (-6 * xi + 6 * xi**2)               # Node 1: d(u_y)/dxi
        dN8_dxi  = (6 * xi - 6 * xi**2)                # Node 2: d(u_y)/dxi

        d2N2_dxi2 = -6 + 12 * xi                       # Node 1: d²(u_y)/dxi²
        d2N8_dxi2 = 6 - 12 * xi                        # Node 2: d²(u_y)/dxi²

        # Rotation shape functions (θ_z --> N6, N12)

        N6  = (L / 2) * (xi - 2 * xi**2 + xi**3)       # Node 1: θ_z
        N12 = (L / 2) * (-xi**2 + xi**3)               # Node 2: θ_z

        dN6_dxi  = (L / 2) * (1 - 4 * xi + 3 * xi**2)  # Node 1: d(θ_z)/dxi
        dN12_dxi = (L / 2) * (-2 * xi + 3 * xi**2)     # Node 2: d(θ_z)/dxi

        d2N6_dxi2 = (L / 2) * (-4 + 6 * xi)            # Node 1: d²(θ_z)/dxi²
        d2N12_dxi2 = (L / 2) * (-2 + 6 * xi)           # Node 2: d²(θ_z)/dxi²

        # ===================================================================
        # 3. Bending in XZ Plane (Hermite Cubic Functions)
        # ===================================================================

        # Translation shape functions (u_z --> N3, N9)

        N3  = 1 - 3 * xi**2 + 2 * xi**3             # Node 1: u_z
        N9  = 3 * xi**2 - 2 * xi**3                 # Node 2: u_z

        dN3_dxi  = (-6 * xi + 6 * xi**2)            # Node 1: d(u_z)/dxi
        dN9_dxi  = (6 * xi - 6 * xi**2)             # Node 2: d(u_z)/dxi

        d2N3_dxi2 = -6 + 12 * xi                    # Node 1: d²(u_z)/dxi²
        d2N9_dxi2 = 6 - 12 * xi                     # Node 2: d²(u_z)/dxi²

        # Rotation shape functions (θ_y --> N5, N11)

        N5  = -(L / 2) * (xi - 2 * xi**2 + xi**3)   # Node 1: θ_y (negative due to rotation direction)
        N11 = -(L / 2) * (-xi**2 + xi**3)           # Node 2: θ_y (negative due to rotation direction)

        dN5_dxi  = -(L / 2) * (1 - 4 * xi + 3 * xi**2)  # Node 1: d(θ_y)/dxi
        dN11_dxi = -(L / 2) * (-2 * xi + 3 * xi**2)     # Node 2: d(θ_y)/dxi

        d2N5_dxi2 = -(L / 2) * (-4 + 6 * xi)        # Node 1: d²(θ_y)/dxi²
        d2N11_dxi2 = -(L / 2) * (-2 + 6 * xi)       # Node 2: d²(θ_y)/dxi²

        # ===================================================================
        # 4. Torsion Shape Functions (Linear Interpolation Functions)
        # ===================================================================

        # Rotation shape functions (θ_y --> N5, N11)

        N4 = 0.5 * (1 - xi)                            # Node 1: θ_x
        N10 = 0.5 * (1 + xi)                           # Node 2: θ_x

        dN4_dxi = -0.5 * np.ones(g)                    # Node 1: d(θ_x)/dxi
        dN10_dxi = 0.5 * np.ones(g)                    # Node 2: d(θ_x)/dxi
 
        d2N4_dxi2 = np.zeros(g)                        # Node 1: d²(θ_x)/dxi²
        d2N10_dxi2 = np.zeros(g)                       # Node 2: d²(θ_x)/dxi²

        # ===================================================================
        #               ASSEMBLE SHAPE FUNCTION MATRICES                 
        # ===================================================================

        N_matrix = np.zeros((g, 12, 6))
        dN_dxi_matrix = np.zeros((g, 12, 6))
        d2N_dxi2_matrix = np.zeros((g, 12, 6))

        # ===================================================================
        # DOF Index → Physical Interpretation → Shape Function
        # ===================================================================
        
        # Node 1 (Indices 0 1 2 3 4 5):
        #   0 => u_x  => N1   (Axial)
        #   1 => u_y  => N2   (Bending in XY plane)
        #   2 => u_z  => N3   (Bending in XZ plane)
        #   3 => θ_x  => N4   (Torsion)
        #   4 => θ_y  => N5   (Bending in XZ plane)
        #   5 => θ_z  => N6   (Bending in XY plane)
        
        # Node 2 (Indices 6 7 8 9 10 11):
        #   6  => u_x => N7   (Axial)
        #   7  => u_y => N8   (Bending in XY plane)
        #   8  => u_z => N9   (Bending in XZ plane)
        #   9  => θ_x => N10  (Torsion)
        #   10 => θ_y => N11  (Bending in XZ plane)
        #   11 => θ_z => N12  (Bending in XY plane)

        # ===================================================================
        # 1. N_matrix: Shape function values
        # ===================================================================

        # -- Node 1 --
        N_matrix[:, 0, 0] = N1                       # u_x @ Node 1
        N_matrix[:, 1, 1] = N2                       # u_y @ Node 1
        N_matrix[:, 2, 2] = N3                       # u_z @ Node 1
        N_matrix[:, 3, 3] = N4                       # θ_x @ Node 1
        N_matrix[:, 4, 4] = N5                       # θ_y @ Node 1
        N_matrix[:, 5, 5] = N6                       # θ_z @ Node 1

        # -- Node 2 --
        N_matrix[:,  6, 0] = N7                      # u_x @ Node 2
        N_matrix[:,  7, 1] = N8                      # u_y @ Node 2
        N_matrix[:,  8, 2] = N9                      # u_z @ Node 2
        N_matrix[:,  9, 3] = N10                     # θ_x @ Node 2
        N_matrix[:, 10, 4] = N11                     # θ_y @ Node 2
        N_matrix[:, 11, 5] = N12                     # θ_z @ Node 2

        # ===================================================================
        # 2. dN_dxi_matrix: Shape function values
        # ===================================================================

        # Axial Strain (ε_x = du_x/dx) → Axial deformation along beam length**
        # Torsional Strain (γ_x = dθ_x/dx, twist about X-axis)**

        # -- Node 1 --
        dN_dxi_matrix[:, 0, 0] = dN1_dxi             # du_x/dxi @ Node 1
        dN_dxi_matrix[:, 1, 1] = dN2_dxi             # du_y/dxi @ Node 1
        dN_dxi_matrix[:, 2, 2] = dN3_dxi             # du_z/dxi @ Node 1
        dN_dxi_matrix[:, 3, 3] = dN4_dxi             # dθ_x/dxi @ Node 1
        dN_dxi_matrix[:, 4, 4] = dN5_dxi             # dθ_y/dxi @ Node 1
        dN_dxi_matrix[:, 5, 5] = dN6_dxi             # dθ_z/dxi @ Node 1

        # -- Node 2 --
        dN_dxi_matrix[:,  6, 0] = dN7_dxi            # du_x/dxi @ Node 2
        dN_dxi_matrix[:,  7, 1] = dN8_dxi            # du_y/dxi @ Node 2
        dN_dxi_matrix[:,  8, 2] = dN9_dxi            # du_z/dxi @ Node 2
        dN_dxi_matrix[:,  9, 3] = dN10_dxi           # dθ_x/dxi @ Node 2
        dN_dxi_matrix[:, 10, 4] = dN11_dxi           # dθ_y/dxi @ Node 2
        dN_dxi_matrix[:, 11, 5] = dN12_dxi           # dθ_z/dxi @ Node 2

        # ===================================================================
        # 3. d2N_dxi2_matrix: Shape function values
        # ===================================================================

        # Bending in XZ Plane (κ_z = d²u_y/dx², bending about Z-axis)**
        # Bending in XY Plane (κ_y = d²u_z/dx², bending about Y-axis)**

        # -- Node 1 --
        d2N_dxi2_matrix[:, 0, 0] = d2N1_dxi2         # d²u_x/dxi² @ Node 1
        d2N_dxi2_matrix[:, 1, 1] = d2N2_dxi2         # d²u_y/dxi² @ Node 1
        d2N_dxi2_matrix[:, 2, 2] = d2N3_dxi2         # d²u_z/dxi² @ Node 1
        d2N_dxi2_matrix[:, 3, 3] = d2N4_dxi2         # d²θ_x/dxi² @ Node 1
        d2N_dxi2_matrix[:, 4, 4] = d2N5_dxi2         # d²θ_y/dxi² @ Node 1
        d2N_dxi2_matrix[:, 5, 5] = d2N6_dxi2         # d²θ_z/dxi² @ Node 1

        # -- Node 2 --
        d2N_dxi2_matrix[:,  6, 0] = d2N7_dxi2        # d²u_x/dxi² @ Node 2
        d2N_dxi2_matrix[:,  7, 1] = d2N8_dxi2        # d²u_y/dxi² @ Node 2
        d2N_dxi2_matrix[:,  8, 2] = d2N9_dxi2        # d²u_z/dxi² @ Node 2
        d2N_dxi2_matrix[:,  9, 3] = d2N10_dxi2       # d²θ_x/dxi² @ Node 2
        d2N_dxi2_matrix[:, 10, 4] = d2N11_dxi2       # d²θ_y/dxi² @ Node 2
        d2N_dxi2_matrix[:, 11, 5] = d2N12_dxi2       # d²θ_z/dxi² @ Node 2

        return N_matrix, dN_dxi_matrix, d2N_dxi2_matrix
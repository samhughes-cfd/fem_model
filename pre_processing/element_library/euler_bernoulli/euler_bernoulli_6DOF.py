# pre_processing\element_library\euler_bernoulli\euler_bernoulli_6DOF.py

import os
import numpy as np
import logging
from scipy.sparse import coo_matrix
from typing import Tuple
from pre_processing.element_library.element_1D_base import Element1DBase
from pre_processing.element_library.utilities.interpolate_loads import interpolate_loads

class EulerBernoulliBeamElement6DOF(Element1DBase):
    """
    6-DOF 3D Euler-Bernoulli Beam Element with full matrix computation capabilities
    
    Features:
    - Exact shape function implementation
    - Configurable quadrature order
    - Combined point/distributed load handling
    - Property-based access to material/geometry parameters
    """
    
    def __init__(self, 
                 geometry_array: np.ndarray,
                 material_array: np.ndarray,
                 mesh_dictionary: dict,
                 point_load_array: np.ndarray,
                 distributed_load_array: np.ndarray,
                 element_id: int,
                 quadrature_order: int = 3):
        """
        Initialize a 6-DOF beam element

        Args:
            geometry_array: Geometry properties array [1x20]
            material_array: Material properties array [1x4]
            mesh_dictionary: Mesh data dictionary
            point_load_array: Point load array [Nx9]
            distributed_load_array: Distributed load array [Nx9]
            element_id: Element ID in the mesh
            quadrature_order: Integration order (default=3)
        """
        super().__init__(geometry_array, material_array, mesh_dictionary,
                         point_load_array, distributed_load_array, dof_per_node=6)
        
        self.element_id = element_id
        self.quadrature_order = quadrature_order
        
        # Initialize element geometry
        self._init_element_geometry()
        self._validate_element_properties()

    def _init_element_geometry(self) -> None:
        """Initialize element-specific geometric properties"""
        conn = self.mesh_dictionary["connectivity"][self.element_id]
        self.node_coords = self.mesh_dictionary["node_coordinates"][conn]
        
        self.x_start = self.node_coords[0, 0]
        self.x_end = self.node_coords[1, 0]
        self.L = self.x_end - self.x_start
        
        # Get global X maximum for boundary condition handling
        self.x_global_end = np.max(self.mesh_dictionary["node_coordinates"][:, 0])

    def _validate_element_properties(self) -> None:
        """Validate critical element properties"""
        if self.L <= 0:
            raise ValueError(f"Invalid element length {self.L:.2e} for element {self.element_id}")
        if not hasattr(self, 'material_array') or self.material_array.size == 0:
            raise ValueError("Material properties not properly initialized")
        if self.geometry_array.shape != (1, 20):
            raise ValueError("Geometry array must have shape (1, 20)")

    # Property definitions -----------------------------------------------------
    @property
    def A(self) -> float:
        """Cross-sectional area (m²)"""
        return self.geometry_array[0, 1]
    
    @property
    def I_x(self) -> float:
        """Torsional moment of inertia (m⁴)"""
        return self.geometry_array[0, 2]
    
    @property
    def I_y(self) -> float:
        """Moment of inertia about y-axis (m⁴)"""
        return self.geometry_array[0, 3]
    
    @property
    def I_z(self) -> float:
        """Moment of inertia about z-axis (m⁴)"""
        return self.geometry_array[0, 4]
    
    @property
    def E(self) -> float:
        """Young's modulus (Pa)"""
        return self.material_array[0, 0]
    
    @property
    def G(self) -> float:
        """Shear modulus (Pa)"""
        return self.material_array[0, 1]
    
    @property
    def integration_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gauss quadrature points/weights"""
        return np.polynomial.legendre.leggauss(self.quadrature_order)

    # Shape functions ----------------------------------------------------------
    def shape_functions(self, xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        N3 = 0.25 * L * (1 - xi - xi**2 + xi**3)  # Rotation
        N9 = 0.25 * L * (1 + xi - xi**2 - xi**3)  # Rotation

        # **Derivatives**
        dN2_dxi = -3 * xi + 2 * xi**2
        dN8_dxi = 3 * xi - 2 * xi**2
        dN3_dxi = 0.25 * L * (1 - 4 * xi + 3 * xi**2)
        dN9_dxi = 0.25 * L * (-2 * xi + 3 * xi**2)

        d2N2_dxi2 = -3 + 4 * xi
        d2N8_dxi2 = 3 - 4 * xi
        d2N3_dxi2 = 0.25 * L * (-4 + 6 * xi)
        d2N9_dxi2 = 0.25 * L * (-2 + 6 * xi)

        # **3. Bending in XZ Plane (Reuse Hermite Cubic Functions)**
        N4, N10 = N2, N8  # Displacement
        N5, N11 = N3, N9  # Rotation

        # **Derivatives**
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
        N_matrix[:, 4, 4] = N3   # Node 1 rotation (θ_y)
        N_matrix[:, 10, 4] = N9  # Node 2 rotation (θ_y)

        ### **Rotation DOF (θ_z - rotation about Z-axis, associated with u_y)**
        # Hermite cubic shape functions for bending rotation in the XZ plane
        N_matrix[:, 5, 5] = N5   # Node 1 rotation (θ_z)
        N_matrix[:, 11, 5] = N11 # Node 2 rotation (θ_z)

        # ---------------------------------------------------------------------------

        # **First Derivative Assignments (Strains & Curvatures)**

        ### **Axial Strain (ε_x = du_x/dx) → Axial deformation along beam length**
        dN_dxi_matrix[:, 0, 0] = dN1_dxi  # Node 1 axial strain (du_x/dx)
        dN_dxi_matrix[:, 6, 0] = dN7_dxi  # Node 2 axial strain (du_x/dx)

        # --------------------------------------------------------------------

        ### **Bending in XZ Plane (κ_z = d²u_y/dx², bending about Z-axis)**
        # Transverse displacement u_y bending along the Z-axis
        dN_dxi_matrix[:, 1, 1] = dN2_dxi   # Node 1 bending strain (du_y/dx)
        dN_dxi_matrix[:, 7, 1] = dN8_dxi   # Node 2 bending strain (du_y/dx)

        ### **Bending in XY Plane (κ_y = d²u_z/dx², bending about Y-axis)**
        # Transverse displacement u_z bending along the Y-axis
        dN_dxi_matrix[:, 2, 2] = dN4_dxi   # Node 1 bending strain (du_z/dx)
        dN_dxi_matrix[:, 8, 2] = dN10_dxi  # Node 2 bending strain (du_z/dx)

        # --------------------------------------------------------------------

        ### **Torsional Strain (γ_x = dθ_x/dx, twist about X-axis)**
        dN_dxi_matrix[:, 3, 3] = dN6_dxi   # Node 1 torsional strain (dθ_x/dx)
        dN_dxi_matrix[:, 9, 3] = dN12_dxi  # Node 2 torsional strain (dθ_x/dx)

        # --------------------------------------------------------------------

        ### **Rotation in XY Plane (dθ_y/dx, bending about the Y-axis)**
        # Associated with u_z displacement (bending in the XY plane)
        dN_dxi_matrix[:, 4, 4] = dN3_dxi   # Node 1 bending rotation (dθ_y/dx)
        dN_dxi_matrix[:, 10, 4] = dN9_dxi  # Node 2 bending rotation (dθ_y/dx)

        ### **Rotation in XZ Plane (dθ_z/dx, bending about the Z-axis)**
        # Associated with u_y displacement (bending in the XZ plane)
        dN_dxi_matrix[:, 5, 5] = dN5_dxi   # Node 1 bending rotation (dθ_z/dx)
        dN_dxi_matrix[:, 11, 5] = dN11_dxi # Node 2 bending rotation (dθ_z/dx)


        # ---------------------------------------------------------------------------

        # **Second Derivative Assignments (For Bending Moments Only)**
        # Second derivatives are only relevant for bending (curvatures).

        ### **Bending in XZ Plane (κ_z = d²u_y/dx², bending about Z-axis)**
        d2N_dxi2_matrix[:, 1, 1] = d2N2_dxi2   # Node 1 curvature (u_y)
        d2N_dxi2_matrix[:, 7, 1] = d2N8_dxi2   # Node 2 curvature (u_y)

        ### **Bending in XY Plane (κ_y = d²u_z/dx², bending about Y-axis)**
        d2N_dxi2_matrix[:, 2, 2] = d2N4_dxi2   # Node 1 curvature (u_z)
        d2N_dxi2_matrix[:, 8, 2] = d2N10_dxi2  # Node 2 curvature (u_z)

        # --------------------------------------------------------------------

        ### **Rotation in XY Plane (d²θ_y/dx², bending about the Y-axis)**
        # Associated with u_z displacement (bending in the XY plane)
        d2N_dxi2_matrix[:, 4, 4] = d2N5_dxi2    # Node 1 rotation (θ_y)
        d2N_dxi2_matrix[:, 10, 4] = d2N11_dxi2  # Node 2 rotation (θ_y)

        ### **Rotation in XZ Plane (d²θ_z/dx², bending about the Z-axis)**
        # Associated with u_y displacement (bending in the XZ plane)
        d2N_dxi2_matrix[:, 5, 5] = d2N3_dxi2    # Node 1 rotation (θ_z)
        d2N_dxi2_matrix[:, 11, 5] = d2N9_dxi2   # Node 2 rotation (θ_z)

        # --------------------------------------------------------------------

        ### **Torsional Second Derivatives (γ_x = d²θ_x/dx²)**
        # Associated with the twisting of the beam about its longitudinal axis (x-axis)
        d2N_dxi2_matrix[:, 3, 3] = d2N6_dxi2    # Node 1 torsional stiffness
        d2N_dxi2_matrix[:, 9, 3] = d2N12_dxi2   # Node 2 torsional stiffness

        return N_matrix, dN_dxi_matrix, d2N_dxi2_matrix

    # Tensor computations ------------------------------------------------------
    def element_stiffness_matrix(self, job_results_dir: str = None) -> np.ndarray:
        """Compute the element stiffness matrix with improved robustness and detailed logging."""
        # Configure logging for stiffness matrix computations
        self.configure_element_stiffness_logging(job_results_dir)
    
        xi_points, weights = self.integration_points  # Gauss points & weights
        detJ = self.L / 2  # Jacobian determinant
    
        # ✅ **Material stiffness matrix (D)**
        D = np.diag([
            self.E * self.A,  # Axial stiffness
            self.E * self.I_z,  # Bending about Z-axis
            self.E * self.I_y,  # Bending about Y-axis
            self.G * self.I_x   # Torsion stiffness
        ])
    
        Ke = np.zeros((12, 12))  # Initialize global stiffness matrix

        # Log initialization
        self.logger.debug(f"Element Stiffness Matrix Computation (Element ID: {self.element_id})")
        self.logger.debug(f"DetJ: {detJ:.6e}")
        self.logger.debug("Material Stiffness Matrix (D):")
        self.logger.debug(np.array2string(D, precision=6, suppress_small=True))

        # =====================================================
        # ✅ **Loop Over Gauss Points**
        # =====================================================
        for g, (xi_g, w_g) in enumerate(zip(xi_points, weights)):
            # ✅ **Get shape function derivatives at Gauss point**
            _, dN_dxi, d2N_dxi2 = self.shape_functions(xi_g)

            # ✅ **Initialize B-matrix (Strain-Displacement Matrix)**
            B = np.zeros((4, 12))

            # -------------------------------------------------------------
            # **1️⃣ Axial Strain (ε_x = du_x/dx)**
            # - Affects axial displacement (u_x)
            # - Uses first derivatives of shape functions
            # -------------------------------------------------------------
            B[0, 0] = dN_dxi[0, 0, 0] / detJ  # Node 1 (u_x)
            B[0, 6] = dN_dxi[0, 6, 0] / detJ  # Node 2 (u_x)

            # -------------------------------------------------------------
            # **2️⃣ Bending in XZ Plane (κ_z = d²u_y/dx²)**
            # - Affects transverse displacement u_y (bending about Z-axis)
            # - Uses second derivatives of shape functions
            # -------------------------------------------------------------
            B[1, 1] = d2N_dxi2[0, 1, 1] / detJ ** 2  # Node 1 (u_y)
            B[1, 7] = d2N_dxi2[0, 7, 1] / detJ ** 2  # Node 2 (u_y)

            # -------------------------------------------------------------
            # **3️⃣ Bending in XY Plane (κ_y = d²u_z/dx²)**
            # - Affects transverse displacement u_z (bending about Y-axis)
            # - Uses second derivatives of shape functions
            # -------------------------------------------------------------
            B[2, 2] = d2N_dxi2[0, 2, 2] / detJ ** 2  # Node 1 (u_z)
            B[2, 8] = d2N_dxi2[0, 8, 2] / detJ ** 2  # Node 2 (u_z)

            # -------------------------------------------------------------
            # **4️⃣ Torsional Strain (γ_x = dθ_x/dx)**
            # - Affects rotation about X-axis (torsional twist)
            # - Uses first derivatives of shape functions
            # -------------------------------------------------------------
            B[3, 3] = dN_dxi[0, 3, 3] / detJ  # Node 1 (θ_x)
            B[3, 9] = dN_dxi[0, 9, 3] / detJ  # Node 2 (θ_x)

            # -------------------------------------------------------------
            # **5️⃣ Rotation about Y-axis (d²θ_y/dx²) → Bending in XY plane**
            # - Associated with u_z displacement
            # - Uses second derivatives of shape functions
            # -------------------------------------------------------------
            B[1, 4] = d2N_dxi2[0, 4, 4] / detJ ** 2  # Node 1 (θ_y)
            B[1, 10] = d2N_dxi2[0, 10, 4] / detJ ** 2  # Node 2 (θ_y)

            # -------------------------------------------------------------
            # **6️⃣ Rotation about Z-axis (d²θ_z/dx²) → Bending in XZ plane**
            # - Associated with u_y displacement
            # - Uses second derivatives of shape functions
            # -------------------------------------------------------------
            B[2, 5] = d2N_dxi2[0, 5, 5] / detJ ** 2  # Node 1 (θ_z)
            B[2, 11] = d2N_dxi2[0, 11, 5] / detJ ** 2  # Node 2 (θ_z)

            # ✅ **Compute stiffness contribution**
            Ke_contribution = np.einsum('ij, jk, kl -> il', B.T, D, B) * w_g * detJ
            Ke += Ke_contribution

            # Log Gauss point computations
            self.logger.debug(f"\n----- Gauss Point {g + 1}/{len(xi_points)} -----")
            self.logger.debug(f"ξ (xi): {xi_g:.6e}, Weight: {w_g:.6e}")
            self.logger.debug("Shape Function First Derivatives (dN_dxi):")
            self.logger.debug(np.array2string(dN_dxi[0], precision=6, suppress_small=True))
            self.logger.debug("Shape Function Second Derivatives (d2N_dxi2):")
            self.logger.debug(np.array2string(d2N_dxi2[0], precision=6, suppress_small=True))
            self.logger.debug("Strain-Displacement Matrix (B):")
            self.logger.debug(np.array2string(B, precision=6, suppress_small=True))
            self.logger.debug("Strain-Displacement Matrix Transpose (B.T):")
            self.logger.debug(np.array2string(B.T, precision=6, suppress_small=True))
            self.logger.debug("Ke Contribution at Gauss Point:")
            self.logger.debug(np.array2string(Ke_contribution, precision=6, suppress_small=True))
            self.logger.debug("")  # Add a space between Gauss point evaluations

        # Final logging summary
        self.logger.debug("\n===== Final Element Stiffness Matrix =====")
        self.logger.debug(np.array2string(Ke, precision=6, suppress_small=True))

        return Ke

    def element_force_vector(self, job_results_dir: str = None) -> np.ndarray:
        """Compute the element force vector with improved robustness and detailed logging."""
        # Configure logging for force vector computations
        self.configure_element_force_logging(job_results_dir)

        Fe = np.zeros(12)  # Initialize element force vector

        # ✅ **Log initialization**
        self.logger.debug(f"Element Force Vector Computation (Element ID: {self.element_id})")

        # =====================================================
        # ✅ **1. Process Distributed Loads (Body Forces)**
        # =====================================================
        if self.distributed_load_array.size > 0:
            xi_gauss, weights = self.integration_points  # Gauss quadrature points & weights
            x_gauss = (xi_gauss + 1) * (self.L / 2) + self.x_start  # Convert xi to global x

            # ✅ **Interpolate distributed loads at Gauss points**
            q_gauss = interpolate_loads(x_gauss, self.distributed_load_array)

            # ✅ **Evaluate shape functions at Gauss points**
            N, _, _ = self.shape_functions(xi_gauss)

            # ✅ **Compute force contribution from distributed loads**
            Fe_dist = np.einsum("gij,gj,g->i", N, q_gauss, weights) * (self.L / 2)
            Fe += Fe_dist  # Accumulate into total force vector

            # Log distributed load integration
            self.logger.debug("\n===== Distributed Load Contribution =====")
            for g, (xi_g, w_g) in enumerate(zip(xi_gauss, weights)):
                self.logger.debug(f"\n----- Gauss Point {g + 1}/{len(xi_gauss)} -----")
                self.logger.debug(f"ξ (xi): {xi_g:.6e}, Weight: {w_g:.6e}")
                self.logger.debug("Shape Function Values (N):")
                self.logger.debug(np.array2string(N[g], precision=6, suppress_small=True))
                self.logger.debug("Interpolated Load Values (q_gauss):")
                self.logger.debug(np.array2string(q_gauss[g], precision=6, suppress_small=True))
                self.logger.debug("Force Contribution from Distributed Load:")
                self.logger.debug(np.array2string(Fe_dist.reshape(1, -1), precision=6, suppress_small=True))

        # =====================================================
        # ✅ **2. Process Point Loads**
        # =====================================================
        if self.point_load_array.size > 0:
            for load in self.point_load_array:
                x_p = load[0]  # Load position (global x-coordinate)
                F_p = load[3:9]  # Load vector (Fx, Fy, Fz, Mx, My, Mz)

                # ✅ **Ensure load is within element bounds**
                if (self.x_start <= x_p <= self.x_end) if np.isclose(self.x_end, self.x_global_end) \
                    else (self.x_start <= x_p < self.x_end):

                    xi_p = 2 * (x_p - self.x_start) / self.L - 1  # Convert to local coordinates

                    # ✅ **Evaluate shape functions at load location**
                    N_p, _, _ = self.shape_functions(np.array([xi_p]))

                    # ✅ **Separate translational (force) and rotational (moment) contributions**
                    Fe_trans = np.einsum("ij,j->i", N_p[0, :, :3], F_p[:3])  # Translational (Fx, Fy, Fz)
                    Fe_rot = np.einsum("ij,j->i", N_p[0, :, 3:], F_p[3:])  # Rotational moments (Mx, My, Mz)

                    # ✅ **Correct indexing for proper DOF allocation**
                    Fe[[0, 1, 2, 6, 7, 8]] += Fe_trans  # Apply translational contributions (u_x, u_y, u_z)
                    Fe[[3, 4, 5, 9, 10, 11]] += Fe_rot  # Apply rotational contributions (θ_x, θ_y, θ_z)

                    ## Enhanced debug logging for validation
                    self.logger.debug(f"\nPoint Load at x_p = {x_p:.6e} (xi_p = {xi_p:.6e})")
                    self.logger.debug(f"Point Load Vector (F_p): {F_p}")
                    self.logger.debug("Shape Function Values at Load Point:")
                    self.logger.debug(np.array2string(N_p[0], precision=6, suppress_small=True))
                    self.logger.debug("Separated Translational Contribution:")
                    self.logger.debug(np.array2string(Fe_trans.reshape(1, -1), precision=6, suppress_small=True))
                    self.logger.debug("Separated Rotational Contribution:")
                    self.logger.debug(np.array2string(Fe_rot.reshape(1, -1), precision=6, suppress_small=True))
                    self.logger.debug("Updated Element Force Vector:")
                    self.logger.debug(np.array2string(Fe.reshape(1, -1), precision=6, suppress_small=True))

        # Final log summary
        self.logger.debug("\n===== Final Element Force Vector =====")
        self.logger.debug(np.array2string(Fe.reshape(1, -1), precision=6, suppress_small=True))

        return Fe

    # Additional functionality -------------------------------------------------
    @property
    def connectivity(self) -> np.ndarray:
        """Element node connectivity"""
        return self.mesh_dictionary["connectivity"][self.element_id]

    def __repr__(self) -> str:
        return (f"EulerBernoulliBeamElement6DOF(element_id={self.element_id}, "
                f"L={self.L:.2e}m, E={self.E:.2e}Pa, "
                f"quad_order={self.quadrature_order})")
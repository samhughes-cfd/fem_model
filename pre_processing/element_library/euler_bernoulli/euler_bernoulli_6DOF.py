# pre_processing\element_library\euler_bernoulli\euler_bernoulli_6DOF.py

import os
import numpy as np
import logging
from scipy.sparse import coo_matrix
from typing import Tuple
from pre_processing.element_library.element_1D_base import Element1DBase
from pre_processing.element_library.euler_bernoulli.utilities.shape_functions_6DOF import shape_functions
from pre_processing.element_library.euler_bernoulli.utilities.D_matrix_6DOF import D_matrix
from pre_processing.element_library.euler_bernoulli.utilities.B_matrix_6DOF import B_matrix
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

    # Wrapper functions calling external utilities ------------------------------------

    def shape_functions(self, xi: np.ndarray):
        return shape_functions(xi, self.L)

    def D_matrix(self):
        return D_matrix(self.E, self.G, self.A, self.I_y, self.I_z, self.I_x)

    def B_matrix(self, dN_dxi: np.ndarray, d2N_dxi2: np.ndarray):
        return B_matrix(dN_dxi, d2N_dxi2, self.L)

    # Tensor computations ------------------------------------------------------
    def element_stiffness_matrix(self, job_results_dir: str = None) -> np.ndarray:
        """
        Compute the Euler-Bernoulli 3D Beam Element stiffness matrix.
    
        Features:
        - Modular clarity with external utilities (B_matrix, D_matrix, shape_functions)
        - Detailed per-Gauss-point logging for debugging and verification purposes
    
        Args:
            job_results_dir (str, optional): Directory path for logging outputs.
    
        Returns:
            np.ndarray: Element stiffness matrix [12x12].
        """
        # Configure detailed element-level logging
        self.configure_element_stiffness_logging(job_results_dir)

        xi, w = self.integration_points
        detJ = self.L / 2
        Ke = np.zeros((12, 12))

        # Assemble material stiffness matrix (D)
        D = self.D_matrix()
    
        # Initial Logging
        self.logger.debug(f"Element Stiffness Matrix Computation (Element ID: {self.element_id})")
        self.logger.debug(f"Element Length (L): {self.L:.6e}, Jacobian determinant (detJ): {detJ:.6e}")
        self.logger.debug("Material Stiffness Matrix (D):")
        self.logger.debug(np.array2string(D, precision=6, suppress_small=True))

        # Gauss Quadrature Integration
        for g, (xi_g, w_g) in enumerate(zip(xi, w)):
        
            # Evaluate shape function derivatives at the current Gauss point
            _, dN_dxi, d2N_dxi2 = self.shape_functions(np.array([xi_g]))
        
            # Compute the strain-displacement B-matrix (corrected external call)
            B = self.B_matrix(dN_dxi, d2N_dxi2)[0]

            # Compute stiffness contribution at current Gauss point
            Ke_contribution = B.T @ D @ B * w_g * detJ
            Ke += Ke_contribution

            # ----- Granular Logging for Verification & Debugging -----
            self.logger.debug(f"\n----- Gauss Point {g + 1}/{len(xi)} -----")
            self.logger.debug(f"Natural Coordinate (xi): {xi_g:.6e}, Gauss Weight: {w_g:.6e}")

            self.logger.debug("Shape Function First Derivatives (dN_dxi):")
            self.logger.debug(np.array2string(dN_dxi[0], precision=6, suppress_small=True))

            self.logger.debug("Shape Function Second Derivatives (d2N_dxi2):")
            self.logger.debug(np.array2string(d2N_dxi2[0], precision=6, suppress_small=True))

            self.logger.debug("Strain-Displacement Matrix (B):")
            self.logger.debug(np.array2string(B, precision=6, suppress_small=True))

            self.logger.debug("Strain-Displacement Matrix Transpose (B.T):")
            self.logger.debug(np.array2string(B.T, precision=6, suppress_small=True))

            self.logger.debug("Element Stiffness Matrix Contribution at Gauss Point:")
            self.logger.debug(np.array2string(Ke_contribution, precision=6, suppress_small=True))
            self.logger.debug("")  # Blank line for readability between Gauss points

        # ----- Final Element Stiffness Matrix Logging -----
        self.logger.debug("\n===== Final Element Stiffness Matrix (Ke) =====")
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

                    # ✅ **Use einsum for efficient multiplication**
                    Fe_trans = np.einsum("ij,j->i", N_p[0, [0, 1, 2, 6, 7, 8], :3], F_p[:3])  # Translational
                    Fe_rot = np.einsum("ij,j->i", N_p[0, [3, 4, 5, 9, 10, 11], 3:], F_p[3:])  # Rotational

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
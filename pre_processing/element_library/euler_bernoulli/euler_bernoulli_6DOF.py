# pre_processing\element_library\euler_bernoulli\euler_bernoulli_6DOF.py

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
        N3 = 0.5 * L * (xi - 2 * xi**2 + xi**3)  # Rotation
        N9 = 0.5 * L * (-xi**2 + xi**3)  # Rotation

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


    # Matrix computations ------------------------------------------------------
    def element_stiffness_matrix(self) -> np.ndarray:
        """Compute the element stiffness matrix using block matrix formulation with debug print statements"""

        xi_points, weights = self.integration_points
        detJ = self.L / 2  # Jacobian determinant

        # Material stiffness matrix (D)
        D = np.diag([
            self.E * self.A,  # Axial stiffness
            self.E * self.I_z,  # Bending about z-axis
            self.E * self.I_y,  # Bending about y-axis
            self.G * self.I_x   # Torsion stiffness
        ])

        #print(f"Material stiffness matrix D:\n{D}")

        # Initialize full stiffness matrix (12 x 12)
        Ke = np.zeros((12, 12))

        # Initialize block matrices (6x6 each)
        K11 = np.zeros((6, 6))
        K12 = np.zeros((6, 6))
        K21 = np.zeros((6, 6))
        K22 = np.zeros((6, 6))

        for g in range(len(xi_points)):

            # Get shape function derivatives
            _, dN_dxi, d2N_dxi2 = self.shape_functions(xi_points[g])

            # Strain-displacement matrix B (4 strains × 12 DOFs)
            B = np.zeros((4, 12))

            # Axial strain (ε = du_x/dx)
            B[0, 0] = (1.0 / detJ) * dN_dxi[0, 0, 0]  # Node 1, axial DOF
            B[0, 6] = (1.0 / detJ) * dN_dxi[0, 6, 0]  # Node 2, axial DOF

            # Bending in z-direction (κ_z = d²u_y/dx²)
            B[1, 1] = (1.0 / detJ) ** 2 * d2N_dxi2[0, 1, 1]
            B[1, 7] = (1.0 / detJ) ** 2 * d2N_dxi2[0, 7, 1]

            # Bending in y-direction (κ_y = d²u_z/dx²)
            B[2, 2] = (1.0 / detJ) ** 2 * d2N_dxi2[0, 2, 2]
            B[2, 8] = (1.0 / detJ) ** 2 * d2N_dxi2[0, 8, 2]

            # Torsion (dθ_x/dx)
            B[3, 3] = (1.0 / detJ) * dN_dxi[0, 3, 3]
            B[3, 9] = (1.0 / detJ) * dN_dxi[0, 9, 3]

            # Debug prints to verify B matrix dimensions
            #print(f"\nIteration {g}: Gauss point {xi_points[g]}")
            #print(f"B.shape: {B.shape}, Expected: (4, 12)")
            #print(f"B = \n{B}")

            # Compute elemental contribution: Bᵀ * D * B
            try:
                Ke_contribution = np.einsum('ij, jk, kl -> il', B.T, D, B) * weights[g] * detJ

                # Debug prints for einsum calculation
                #print(f"Ke_contribution.shape: {Ke_contribution.shape}, Expected: (12, 12)")
                #print(f"Ke_contribution = \n{Ke_contribution}")

                # Assemble blocks
                K11 += Ke_contribution[:6, :6]
                K12 += Ke_contribution[:6, 6:]
                K21 += Ke_contribution[6:, :6]
                K22 += Ke_contribution[6:, 6:]

            except ValueError as e:
                #print(f"\n❌ ERROR at Gauss point {xi_points[g]}: {e}")
                #print(f"Shapes: B.T={B.T.shape}, D={D.shape}, B={B.shape}")
                raise e  # Re-raise error for debugging

        # Assemble final element stiffness matrix using block structure
        Ke[:6, :6] = K11
        Ke[:6, 6:] = K12
        Ke[6:, :6] = K21
        Ke[6:, 6:] = K22

        # Debug print for final assembled matrix
        #print("\nFinal element stiffness matrix Ke:\n", Ke)

        return Ke

    def element_force_vector(self) -> np.ndarray:
        """Compute the element force vector considering all loads"""
        Fe = np.zeros(12)
        
        # Process distributed loads
        if self.distributed_load_array.size > 0:
            xi_gauss, weights = self.integration_points
            x_gauss = (xi_gauss + 1) * (self.L/2) + self.x_start
            
            # Interpolate loads at Gauss points
            q = interpolate_loads(x_gauss, self.distributed_load_array)
            
            # Integrate using shape functions
            N, _, _ = self.shape_functions(xi_gauss)
            Fe += np.einsum("gij,gj,g->i", N, q, weights) * (self.L/2)
        
        # Process point loads
        if self.point_load_array.size > 0:
            for load in self.point_load_array:
                x_p = load[0]
                F = load[3:9]
                
                # Check load inclusion with boundary handling
                if (self.x_start <= x_p <= self.x_end) if np.isclose(self.x_end, self.x_global_end) \
                   else (self.x_start <= x_p < self.x_end):
                    xi_p = 2 * (x_p - self.x_start)/self.L - 1
                    N_p, _, _ = self.shape_functions(np.array([xi_p]))
                    Fe += np.einsum("ij,j->i", N_p[0], F)

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
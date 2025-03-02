# pre_processing/element_library/euler_bernoulli_beam_6dof.py

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
        Compute shape functions and derivatives for 3D beam
    
        Args:
            xi: Natural coordinates [-1, 1]
        
        Returns:
            (N_matrix, dN_dxi_matrix, d2N_dxi2_matrix) each of shape (g, 12, 6)
        """
        xi = np.atleast_1d(xi)
        g = xi.shape[0]
        L = self.L

        # Calculate shape functions and derivatives
        # Axial functions
        N1 = 0.5 * (1 - xi)
        N7 = 0.5 * (1 + xi)
        dN1_dxi = -0.5 * np.ones(g)
        dN7_dxi = 0.5 * np.ones(g)
        d2N1_dxi2 = np.zeros(g)
        d2N7_dxi2 = np.zeros(g)

        # Bending in XY plane
        N2 = 0.25 * (1 - xi)**2 * (2 + xi)
        N8 = 0.25 * (1 + xi)**2 * (2 - xi)
        N3 = (L / 8) * (1 - xi)**2 * (1 + xi)
        N9 = (L / 8) * (1 + xi)**2 * (1 - xi)
    
        # Derivatives for bending
        dN2_dxi = 0.5*(1 - xi)*(2 + xi) - 0.5*(1 - xi)**2
        dN8_dxi = -0.5*(1 + xi)*(2 - xi) + 0.5*(1 + xi)**2
        dN3_dxi = (L/8)*((1 - xi)**2 - 2*(1 - xi)*(1 + xi))
        dN9_dxi = (L/8)*((1 + xi)**2 - 2*(1 + xi)*(1 - xi))
    
        # Second derivatives
        d2N2_dxi2 = 1.5*xi - 0.5
        d2N8_dxi2 = -1.5*xi + 0.5
        d2N3_dxi2 = (L/8)*(3*xi - 1)
        d2N9_dxi2 = (L/8)*(-3*xi + 1)

        # Bending in XZ plane (reuse XY functions)
        N4, N10 = N2, N8
        N5, N11 = N3, N9
        dN4_dxi, dN10_dxi = dN2_dxi, dN8_dxi
        dN5_dxi, dN11_dxi = dN3_dxi, dN9_dxi
        d2N4_dxi2, d2N10_dxi2 = d2N2_dxi2, d2N8_dxi2
        d2N5_dxi2, d2N11_dxi2 = d2N3_dxi2, d2N9_dxi2

        # Torsion functions
        N6, N12 = N1, N7
        dN6_dxi, dN12_dxi = dN1_dxi, dN7_dxi
        d2N6_dxi2, d2N12_dxi2 = d2N1_dxi2, d2N7_dxi2

        # Assemble shape functions and derivatives into matrices
        N_matrix = np.zeros((g, 12, 6))
        dN_dxi_matrix = np.zeros((g, 12, 6))
        d2N_dxi2_matrix = np.zeros((g, 12, 6))

        # Assign shape functions and derivatives to the correct DOFs
        N_matrix[:, 0, 0] = N1  # Axial DOF at node 1
        N_matrix[:, 6, 0] = N7  # Axial DOF at node 2
        N_matrix[:, 1, 1] = N2  # Bending DOF (u_y) at node 1
        N_matrix[:, 7, 1] = N8  # Bending DOF (u_y) at node 2
        N_matrix[:, 2, 2] = N4  # Bending DOF (u_z) at node 1
        N_matrix[:, 8, 2] = N10  # Bending DOF (u_z) at node 2
        N_matrix[:, 3, 3] = N6  # Torsion DOF at node 1
        N_matrix[:, 9, 3] = N12  # Torsion DOF at node 2
        N_matrix[:, 4, 4] = N5  # Rotation DOF (θ_y) at node 1
        N_matrix[:, 10, 4] = N11  # Rotation DOF (θ_y) at node 2
        N_matrix[:, 5, 5] = N3  # Rotation DOF (θ_z) at node 1
        N_matrix[:, 11, 5] = N9  # Rotation DOF (θ_z) at node 2

        # Assign derivatives
        dN_dxi_matrix[:, 0, 0] = dN1_dxi
        dN_dxi_matrix[:, 6, 0] = dN7_dxi
        dN_dxi_matrix[:, 1, 1] = dN2_dxi
        dN_dxi_matrix[:, 7, 1] = dN8_dxi
        dN_dxi_matrix[:, 2, 2] = dN4_dxi
        dN_dxi_matrix[:, 8, 2] = dN10_dxi
        dN_dxi_matrix[:, 3, 3] = dN6_dxi
        dN_dxi_matrix[:, 9, 3] = dN12_dxi
        dN_dxi_matrix[:, 4, 4] = dN5_dxi
        dN_dxi_matrix[:, 10, 4] = dN11_dxi
        dN_dxi_matrix[:, 5, 5] = dN3_dxi
        dN_dxi_matrix[:, 11, 5] = dN9_dxi

        # Assign second derivatives
        d2N_dxi2_matrix[:, 1, 1] = d2N2_dxi2
        d2N_dxi2_matrix[:, 7, 1] = d2N8_dxi2
        d2N_dxi2_matrix[:, 2, 2] = d2N4_dxi2
        d2N_dxi2_matrix[:, 8, 2] = d2N10_dxi2
        d2N_dxi2_matrix[:, 5, 5] = d2N3_dxi2
        d2N_dxi2_matrix[:, 11, 5] = d2N9_dxi2

        return N_matrix, dN_dxi_matrix, d2N_dxi2_matrix

    # Matrix computations ------------------------------------------------------
    def element_stiffness_matrix(self):
        """Compute the element stiffness matrix using Gauss quadrature"""
        xi_points, weights = self.integration_points
        detJ = self.L / 2

        # Material matrix
        D = np.diag([
            self.E * self.A,  # Axial
            self.E * self.I_z,  # Bending Z
            self.E * self.I_y,  # Bending Y
            self.G * self.I_x   # Torsion
        ])

        # Initialize stiffness matrix
        Ke = np.zeros((12, 12))

        for g in range(len(xi_points)):
            
            try:
                # Get shape function derivatives
                _, dN_dxi, d2N_dxi2 = self.shape_functions(xi_points[g])

                # Strain-displacement matrix (4 strains × 12 DOFs)
                B = np.zeros((4, 12))

                # Axial: ε = du_x/dx
                B[0, 0] = (1.0 / detJ) * dN_dxi[0, 0, 0]  # Node 1, axial DOF
                B[0, 6] = (1.0 / detJ) * dN_dxi[0, 6, 0]  # Node 2, axial DOF

                # Bending Z: κ_z = d²u_y/dx²
                B[1, 1] = (1.0 / detJ) ** 2 * d2N_dxi2[0, 1, 1]
                B[1, 7] = (1.0 / detJ) ** 2 * d2N_dxi2[0, 7, 1]

                # Bending Y: κ_y = d²u_z/dx²
                B[2, 2] = (1.0 / detJ) ** 2 * d2N_dxi2[0, 2, 2]
                B[2, 8] = (1.0 / detJ) ** 2 * d2N_dxi2[0, 8, 2]

                # Torsion: dθ_x/dx
                B[3, 3] = (1.0 / detJ) * dN_dxi[0, 3, 3]
                B[3, 9] = (1.0 / detJ) * dN_dxi[0, 9, 3]

                # Log tensor values
                print(f"Iteration {g}: Computing B^T * D")
                print(f"B.shape = {B.shape}, D.shape = {D.shape}")
                print(f"B = \n{B}")
                print(f"D = \n{D}")

                try:
                    # Restrict B to first 4 columns to match D dimensions
                    B_reduced = B[:, :4]

                    print(f"B_reduced.shape = {B_reduced.shape}, D.shape = {D.shape}")
                    print(f"B_reduced = \n{B_reduced}")

                    # Step 1: Compute B^T * D manually
                    BT_D = np.zeros((12, 4))  # Shape (12, 4)
                    for i in range(12):
                        for j in range(4):
                            for k in range(4):
                                BT_D[i, j] += B[i % 4, k] * D[k, j]

                    print(f"BT_D.shape = {BT_D.shape}")
                    print(f"BT_D = \n{BT_D}")

                    # Step 2: Compute BT_D * B manually (extend back to full 12 columns)
                    BT_D_B = np.zeros((12, 12))  # Shape (12, 12)
                    for i in range(12):
                        for j in range(12):
                            for k in range(4):
                                BT_D_B[i, j] += BT_D[i, k] * B[k, j]

                    print(f"BT_D_B.shape = {BT_D_B.shape}")
                    print(f"BT_D_B = \n{BT_D_B}")

                    # Step 3: Scale by weights and Jacobian determinant
                    Ke_contribution = BT_D_B * weights[g] * detJ

                    # Add contribution to stiffness matrix
                    Ke += Ke_contribution

                except Exception as e:
                    print(f"Iteration {g}: Error in einsum computation: {e}")
                    print(f"Shapes: B {B.shape}, D {D.shape}, BT_D N/A")
                    print(f"B = \n{B}")
                    print(f"D = \n{D}")
                    raise e  # Re-raise the exception for debugging
        
            except Exception as e:
                print(f"Fatal error in iteration {g}: {e}")
                print(f"B.shape = {B.shape}, D.shape = {D.shape} (before einsum)")
                print(f"B = \n{B}")
                print(f"D = \n{D}")
                raise e  # Propagate the exception for further debugging
                
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
# pre_processing/element_library/euler_bernoulli/euler_bernoulli_beam_element.py

import numpy as np
from pre_processing.element_library.element_1D_base import Element1DBase
from pre_processing.element_library.utilities.dof_mapping import expand_dof_mapping
from pre_processing.element_library.utilities.shape_function_library.euler_bernoulli_sf import euler_bernoulli_shape_functions
from pre_processing.element_library.utilities.jacobian import compute_jacobian_matrix, compute_jacobian_determinant
from pre_processing.element_library.utilities.gauss_quadrature import integrate_matrix


class EulerBernoulliBeamElement(Element1DBase):
    """
    Euler-Bernoulli beam element (1D) with 6 DOFs per node: 
    (u_x, u_y, u_z, θ_x, θ_y, θ_z).
    Expands to a full 12 DOFs per element system (12x12 stiffness matrix and 12x1 force vector).

    Attributes:
        A (float): Cross-sectional area.
        I_bending_z (float): Second moment of area about the z-axis.
    """

    def __init__(self, element_id, material, section_props, mesh_data, node_positions, loads_array):
        """
        Initializes the Euler-Bernoulli beam element.

        Args:
            element_id (int): Unique identifier for this element.
            material (dict): Material properties dictionary.
            section_props (dict): Section properties dictionary.
            mesh_data (dict): Mesh data dictionary containing connectivity and element lengths.
            node_positions (ndarray): Array of node positions. Shape: (num_nodes, 3)
            loads_array (ndarray): Global loads array. Shape: (num_nodes, 6)
        """
        self.A = section_props["A"]
        self.I_bending_z = section_props["Iz"]  # Moment of inertia about the z-axis

        # Degrees of freedom per node set to 6 for full 3D modeling
        super().__init__(
            element_id=element_id,
            material=material,
            section_props=section_props,
            mesh_data=mesh_data,
            node_positions=node_positions,
            loads_array=loads_array,
            dof_per_node=6
        )

    def shape_functions(self, xi):
        """
        Computes the shape functions and their derivatives for the Euler-Bernoulli beam element.

        Args:
            xi (float): Natural coordinate in [-1, 1].

        Returns:
            tuple: (N, dN_dxi, d2N_dxi2)
                N (ndarray): Shape function vector (6,)
                dN_dxi (ndarray): First derivatives of shape functions w.r.t. xi (6,)
                d2N_dxi2 (ndarray): Second derivatives of shape functions w.r.t. xi (6,)
        """
        return euler_bernoulli_shape_functions(xi, self.get_element_length())

    def material_stiffness_matrix(self):
        """
        Constructs the material stiffness (constitutive) matrix D for the Euler-Bernoulli beam element.

        Returns:
            ndarray: Constitutive matrix D. Shape: (2,2)
        """
        E = self.material["E"]
        A = self.A
        Iz = self.I_bending_z

        # Constitutive matrix for axial (Fx) and bending (Mz) DOFs
        D = np.diag([
            E * A,      # Axial stiffness (Fx)
            E * Iz      # Bending stiffness (Mz)
        ])  # Shape: (2,2)

        return D

    def strain_displacement_matrix(self, dN_dxi):
        """
        Constructs the strain-displacement matrix (B matrix) for the Euler-Bernoulli beam element.

        Args:
            dN_dxi (ndarray): First derivatives of shape functions w.r.t. xi. Shape: (6,)

        Returns:
            ndarray: Strain-displacement matrix. Shape: (2,6)
        """
        # Compute the Jacobian determinant
        node_coords = self.get_node_coordinates()  # Shape: (2, 3)
        jacobian_matrix = compute_jacobian_matrix(
            dN_dxi.reshape(-1, 1),      # Shape: (6,1)
            node_coords.reshape(-1, 1)   # Shape: (2,1)
        )  # Shape: (1,1)
        detJ = compute_jacobian_determinant(jacobian_matrix)  # Scalar

        # Compute dxi/dx
        dxi_dx = 1.0 / detJ  # Scalar

        # Initialize B matrix for axial and bending
        B_axial = np.zeros(6)
        B_axial[0] = dN_dxi[0] * dxi_dx  # Fx_start
        B_axial[3] = dN_dxi[3] * dxi_dx  # Fx_end

        B_bending = np.zeros(6)
        B_bending[1] = dN_dxi[1] * (dxi_dx**2)  # Fy_start
        B_bending[2] = dN_dxi[2] * (dxi_dx**2)  # Mz_start
        B_bending[4] = dN_dxi[4] * (dxi_dx**2)  # Fy_end
        B_bending[5] = dN_dxi[5] * (dxi_dx**2)  # Mz_end

        # Combine axial and bending into B matrix
        B = np.vstack([B_axial, B_bending])  # Shape: (2,6)

        return B

    def element_stiffness_matrix(self):
        """
        Computes the element stiffness matrix and expands it to a 12x12 system.
        Utilizes Gauss quadrature and delegates Jacobian computations to utility functions.
        """
        def integrand_stiffness_matrix(xi):
            """
            Integrand function for stiffness matrix computation at a given natural coordinate xi.

            Args:
                xi (float): Natural coordinate in [-1, 1].

            Returns:
                ndarray: Integrand matrix at xi. Shape: (6,6)
            """
            # Retrieve shape functions and their first and second derivatives
            N, dN_dxi, _ = self.shape_functions(xi)  # Shape: (6,), (6,), (6,)

            # Retrieve nodal coordinates
            node_coords = self.get_node_coordinates()  # Shape: (2,3)

            # Compute Jacobian matrix using utility function
            jacobian_matrix = compute_jacobian_matrix(
                dN_dxi.reshape(-1, 1),      # Shape: (6,1)
                node_coords.reshape(-1, 1)   # Shape: (2,1)
            )  # Shape: (1,1)

            # Compute Jacobian determinant
            detJ = compute_jacobian_determinant(jacobian_matrix)  # Scalar

            # Compute strain-displacement matrix B
            B = self.strain_displacement_matrix(dN_dxi)  # Shape: (2,6)

            # Get material stiffness matrix D
            D = self.material_stiffness_matrix()  # Shape: (2,2)

            # Integrand for stiffness matrix: B.T * D * B * detJ
            integrand = B.T @ D @ B * detJ  # Shape: (6,6)

            return integrand

        # Perform numerical integration using Gauss quadrature with 3 points
        Ke_reduced = integrate_matrix(
            n_gauss=3,
            integrand_func=integrand_stiffness_matrix,
            jacobian_func=lambda xi: compute_jacobian_determinant(
                compute_jacobian_matrix(
                    self.shape_functions(xi)[1].reshape(-1, 1),  # dN_dxi reshaped to (6,1)
                    self.get_node_coordinates().reshape(-1, 1)   # node_coords reshaped to (2,1)
                )
            ),
            dim=1
        )  # Shape: (6,6)

        # Define DOF indices for Fx, Fy, Mz at start and end nodes
        # [Fx_start, Fy_start, Mz_start, Fx_end, Fy_end, Mz_end] correspond to DOFs [0, 1, 5, 6, 7, 11]
        dof_map = [0, 1, 5, 6, 7, 11]

        # Map Ke_reduced to Ke_full using expand_dof_mapping
        Ke_full = expand_dof_mapping(
            reduced_array=Ke_reduced,
            full_size=12,
            dof_map=dof_map
        )  # Shape: (12,12)

        self.Ke = Ke_full  # Shape: (12,12)

    def element_force_vector(self):
        """
        Computes the element force vector by mapping nodal loads to the relevant DOFs.
        Specifically models deformation in indices 0, 1, and 5 (Fx, Fy, Mz).
        Currently handles point (nodal) loads only.
        """
        # Initialize a 12x1 zero vector
        Fe_full = np.zeros(12)

        # Retrieve loads on start and end nodes
        loads = self.get_element_loads()  # Shape: (2,6)

        # Extract Fx, Fy, Mz for start node (node1)
        Fx1, Fy1, _, _, _, Mz1 = loads[0]

        # Extract Fx, Fy, Mz for end node (node2)
        Fx2, Fy2, _, _, _, Mz2 = loads[1]

        # Create reduced force vector [Fx_start, Fy_start, Mz_start, Fx_end, Fy_end, Mz_end]
        Fe_reduced = np.array([Fx1, Fy1, Mz1, Fx2, Fy2, Mz2])

        # Define DOF indices for Fx, Fy, Mz at start and end nodes
        dof_map = [0, 1, 5, 6, 7, 11]  # [Fx_start, Fy_start, Mz_start, Fx_end, Fy_end, Mz_end]

        # Map the reduced force vector to the full force vector using expand_dof_mapping
        Fe_full = expand_dof_mapping(
            reduced_array=Fe_reduced,
            full_size=12,
            dof_map=dof_map
        )  # Shape: (12,)
        
        self.Fe = Fe_full  # Shape: (12,)

    def validate_matrices(self):
        """
        Validates that Ke and Fe have the correct dimensions.

        Raises:
            AssertionError: If Ke or Fe do not have expected dimensions.
        """
        assert self.Ke.shape == (12, 12), f"Ke shape mismatch: Expected (12,12), got {self.Ke.shape}"
        assert self.Fe.shape == (12,), f"Fe shape mismatch: Expected (12,), got {self.Fe.shape}"
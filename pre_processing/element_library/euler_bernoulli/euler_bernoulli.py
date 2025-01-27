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
        I_z (float): Second moment of area about the z-axis.
        E (float): Young's Modulus.
    """

    # Define class-level constants for geometry and material array indices
    GEOMETRY_A_INDEX = 1    # Cross-sectional area (A) in geometry_array
    GEOMETRY_IZ_INDEX = 3   # Moment of inertia about the z-axis (Iz) in geometry_array
    MATERIAL_E_INDEX = 0    # Young's Modulus (E) in material_array

    def __init__(self, element_id: int, material_array: np.ndarray, geometry_array: np.ndarray, 
                 mesh_data: dict, node_positions: np.ndarray, loads_array: np.ndarray):
        """
        Initializes the Euler-Bernoulli beam element.

        Args:
            element_id (int): Unique identifier for this element.
            material_array (np.ndarray): Material properties array of shape (1, 4).
            geometry_array (np.ndarray): Geometry properties array of shape (1, 20).
            mesh_data (dict): Mesh data dictionary containing connectivity, element lengths, and element IDs.
            node_positions (np.ndarray): Array of node positions. Shape: (num_nodes, 3)
            loads_array (np.ndarray): Global loads array. Shape: (N, 9)
        """
        # Extract geometry and material properties using predefined indices
        self.A = geometry_array[0, self.GEOMETRY_A_INDEX]         # Cross-sectional area
        self.I_z = geometry_array[0, self.GEOMETRY_IZ_INDEX]     # Moment of inertia about the z-axis
        self.E = material_array[0, self.MATERIAL_E_INDEX]        # Young's Modulus

        # Initialize the base class with unpacked mesh data
        super().__init__(
            geometry_array=geometry_array,
            material_array=material_array,
            mesh_dictionary=mesh_data,
            loads_array=loads_array,
            dof_per_node=6
        )

    def shape_functions(self, xi: float) -> tuple:
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
        element_index = self.get_element_index()               # Retrieves index based on element_id
        element_length = self.element_lengths_array[element_index]
        return euler_bernoulli_shape_functions(xi, element_length)

    def material_stiffness_matrix(self) -> np.ndarray:
        """
        Constructs the material stiffness (constitutive) matrix D for the Euler-Bernoulli beam element.

        Returns:
            ndarray: Constitutive matrix D. Shape: (2,2)
        """
        E = self.E
        A = self.A
        I_z = self.I_z

        # Constitutive matrix for axial (Fx) and bending (Mz) DOFs
        D = np.diag([
            E * A,      # Axial stiffness (Fx)
            E * I_z     # Bending stiffness (Mz)
        ])  # Shape: (2,2)

        return D

    def strain_displacement_matrix(self, dN_dxi: np.ndarray) -> np.ndarray:
        """
        Constructs the strain-displacement matrix (B matrix) for the Euler-Bernoulli beam element.

        Args:
            dN_dxi (ndarray): First derivatives of shape functions w.r.t. xi. Shape: (6,)

        Returns:
            ndarray: Strain-displacement matrix. Shape: (2,6)
        """
        # Retrieve node indices and coordinates for the current element
        node_indices = self.connectivity_array[self.element_index]  # Node IDs (assuming element_index starts at 0)
        node_coords = self.node_coordinates_array[node_indices - 1]  # Convert node IDs to zero-based indices

        # Compute the Jacobian matrix
        jacobian_matrix = compute_jacobian_matrix(
            dN_dxi.reshape(-1, 1),      # Shape: (6,1)
            node_coords.reshape(-1, 1)   # Shape: (2,1)
        )  # Shape: (1,1)
        detJ = compute_jacobian_determinant(jacobian_matrix)  # Scalar

        if detJ <= 0:
            raise ValueError(f"Invalid Jacobian determinant ({detJ}) for Element ID {self.element_id}. Check node ordering.")

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
        def integrand_stiffness_matrix(xi: float) -> np.ndarray:
            """
            Integrand function for stiffness matrix computation at a given natural coordinate xi.

            Args:
                xi (float): Natural coordinate in [-1, 1].

            Returns:
                ndarray: Integrand matrix at xi. Shape: (6,6)
            """
            # Retrieve shape functions and their first derivatives
            N, dN_dxi, _ = self.shape_functions(xi)  # Shape: (6,), (6,), (6,)

            # Compute strain-displacement matrix B
            B = self.strain_displacement_matrix(dN_dxi)  # Shape: (2,6)

            # Get material stiffness matrix D
            D = self.material_stiffness_matrix()  # Shape: (2,2)

            # Integrand for stiffness matrix: B.T * D * B
            integrand = B.T @ D @ B  # Shape: (6,6)

            return integrand

        # Perform numerical integration using Gauss quadrature with 3 points
        Ke_reduced = integrate_matrix(
            n_gauss=3,
            integrand_func=integrand_stiffness_matrix,
            jacobian_func=lambda xi: compute_jacobian_determinant(
                compute_jacobian_matrix(
                    self.shape_functions(xi)[1].reshape(-1, 1),  # dN_dxi reshaped to (6,1)
                    self.node_coordinates_array[self.connectivity_array[self.element_index] - 1].reshape(-1, 1)   # node_coords reshaped to (2,1)
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

        # Retrieve node indices for the current element
        node_indices = self.connectivity_array[self.element_index]  # Node IDs (e.g., [1, 2])
        node_coords = self.node_coordinates_array[node_indices - 1]  # Shape: (2,3)

        # Retrieve loads for start and end nodes
        # Assuming loads_array has relevant load components mapped appropriately
        # and that self.get_element_loads() returns a (2,6) array with [Fx, Fy, Fz, Mx, My, Mz] per node
        # This method should be defined in the base class
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
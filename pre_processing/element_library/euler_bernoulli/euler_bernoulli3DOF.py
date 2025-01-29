import numpy as np
import pprint
from pre_processing.element_library.element_1D_base import Element1DBase
from pre_processing.element_library.utilities.dof_mapping import expand_dof_mapping
from pre_processing.element_library.utilities.shape_function_library.euler_bernoulli_sf import euler_bernoulli_shape_functions
from pre_processing.element_library.utilities.gauss_quadrature import get_gauss_points
from pre_processing.element_library.utilities.interpolate_loads import interpolate_loads

class EulerBernoulliBeamElement3DOF(Element1DBase):
    """
    Euler-Bernoulli beam element (1D) with 6 DOFs per node:
    (u_x, u_y, u_z, θ_x, θ_y, θ_z).
    """

    # Class-level indices for geometry and material arrays
    GEOMETRY_A_INDEX = 1    # Cross-sectional area (A)
    GEOMETRY_IZ_INDEX = 3   # Moment of inertia about the z-axis (Iz)
    MATERIAL_E_INDEX = 0    # Young's Modulus (E)

    def __init__(self, element_id: int, material_array: np.ndarray, geometry_array: np.ndarray,
                 mesh_dictionary: dict, load_array: np.ndarray):
        """
        Initializes the Euler-Bernoulli beam element.

        Parameters:
        - element_id (int): Unique identifier for the element.
        - material_array (np.ndarray): Material properties array containing Young's modulus.
        - geometry_array (np.ndarray): Geometry properties array containing cross-sectional area and moment of inertia.
        - mesh_dictionary (dict): Mesh data dictionary containing nodal connectivity and coordinates.
        - load_array (np.ndarray): External force/moment distribution on the element.
        """
        # Element properties
        self.element_id = element_id
        self.material_array = material_array
        self.geometry_array = geometry_array
        self.mesh_dictionary = mesh_dictionary
        self.load_array = load_array

        self.element_id = element_id
        self.A = geometry_array[0, self.GEOMETRY_A_INDEX]
        self.I_z = geometry_array[0, self.GEOMETRY_IZ_INDEX]
        self.E = material_array[0, self.MATERIAL_E_INDEX]

        # Compute the Jacobian at initialization
        self.jacobian_matrix, self.detJ = self._compute_jacobian_matrix()

    def get_element_index(self):
        """Finds the index of the element based on its ID."""
        return np.where(self.mesh_dictionary["element_ids"] == self.element_id)[0][0]
    
    def _compute_jacobian_matrix(self):
        """
        Computes the Jacobian matrix and its determinant using shape function derivatives in natural coordinates.

        Returns:
        - jacobian_matrix (np.ndarray): The Jacobian matrix (dx/dxi).
        - jacobian_determinant (float): Determinant of the Jacobian matrix (absolute value).
        """
        # Get the index of the current element
        element_index = self.get_element_index()
    
        # Extract the connectivity for the current element and convert to zero-based indexing
        connectivity = self.mesh_dictionary["connectivity"][element_index] - 1
        # ["connectivity"][element_index] extracts the node IDs that define the current element.  
        # Since node IDs are typically 1-based (ranging from 1 to n), subtracting 1 converts them  
        # to zero-based indexing (ranging from 0 to n-1) to align with Python’s zero-based array indexing.

        # Extract the element length
        element_length = self.mesh_dictionary["element_lengths"][element_index]
    
        # Get the x-coordinates of the nodes in the current element
        node_coords_x = self.mesh_dictionary["node_coordinates"][connectivity][:, 0]
        # ["node_coordinates"][connectivity] extracts the node coordinates [x y z] 
        # of the two nodes defined in the connectivity pair, [:, 0] extracts 
        # all rows : (2 nodes) but strictly the column index 0 (x-coordinate)

        # Compute the shape function derivatives in natural coordinates (at ξ = 0)
        _, dN_dxi, _ = euler_bernoulli_shape_functions(0.0, element_length)

        # Extract only the first and fourth shape function derivatives (indices 0 and 3)
        dN_dxi_specific = np.array([dN_dxi[0], dN_dxi[3]])  # Shape: (2,)

        # Compute the Jacobian matrix (dx/dxi) by multiplying dN_dxi with node coordinates
        jacobian_matrix = np.dot(dN_dxi_specific, node_coords_x)  # Shape: (1, 1) for 1D elements

        # Ensure jacobian_matrix is treated correctly (reshape to 2D if needed)
        jacobian_matrix = np.array([[jacobian_matrix]])  # Explicitly reshape to (1,1) for consistency

        # Compute the determinant of the Jacobian matrix (absolute value to ensure positivity)
        jacobian_determinant = np.abs(jacobian_matrix[0, 0])

        jacobian_verification = element_length / jacobian_determinant

        # Print results
        print("\n===== Jacobian Computation Verification =====")
        pprint.pprint({
            "Jacobian Matrix": jacobian_matrix,
            "Jacobian Determinant": jacobian_determinant,
            "Element Length": element_length,
            "Element Length / Jacobian Determinant": jacobian_verification,
            "Expected Theoretical Value (2)": 2.0
        })
        print("============================================\n")

        return jacobian_matrix, jacobian_determinant
    
    def shape_functions(self, xi: float) -> tuple:
        """
        Evaluates the shape functions and their derivatives at a given natural coordinate ξ.

        Parameters:
        - xi (float): The natural coordinate, in the range [-1, 1].

        Returns:
        - tuple: (shape functions, first derivatives, second derivatives).
        """
        element_index = self.get_element_index()
        element_length = self.mesh_dictionary["element_lengths"][element_index]
        return euler_bernoulli_shape_functions(xi, element_length)

    def material_stiffness_matrix(self) -> np.ndarray:
        """
        Constructs the element material stiffness matrix.

        Returns:
        - np.ndarray: Diagonal matrix representing axial and bending stiffness.
        """
        return np.diag([self.E * self.A, self.E * self.I_z])
        
    def strain_displacement_matrix(self, dN_dxi: np.ndarray) -> np.ndarray:
        """
        Constructs the strain-displacement matrix (B) for the Euler-Bernoulli beam element.

        Parameters:
        - dN_dxi (np.ndarray): The derivatives of shape functions with respect to the natural coordinate ξ.

        Returns:
        - np.ndarray: The strain-displacement matrix B (2x6).
        """
        dxi_dx = 1.0 / self.detJ  # Compute inverse of Jacobian determinant for transformation

        # Initialize axial and bending strain-displacement matrices
        B_axial = np.zeros(6)
        B_axial[0] = dN_dxi[0] * dxi_dx  # Axial strain contribution
        B_axial[3] = dN_dxi[3] * dxi_dx  # Axial strain contribution

        B_bending = np.zeros(6)
        B_bending[1] = dN_dxi[1] * (dxi_dx ** 2)  # Bending curvature
        B_bending[4] = dN_dxi[4] * (dxi_dx ** 2)  # Bending curvature

        # Assemble strain-displacement matrix
        return np.vstack([B_axial, B_bending])  # Shape: (2,6)

    def element_stiffness_matrix(self):
        """
        Computes the element stiffness matrix via numerical integration (Gauss quadrature).

        Returns:
        - np.ndarray: The element stiffness matrix (12x12).
        """
        gauss_points, weights = get_gauss_points(n=3, dim=1)
        shape_data = {xi[0]: self.shape_functions(xi[0]) for xi in gauss_points}
        D = self.material_stiffness_matrix()

        Ke_reduced = sum(
            weights[i] * self._integrand_stiffness_matrix(shape_data, gauss_points[i][0]) * self.detJ
            for i in range(len(weights))
        )

        assert Ke_reduced.shape == (6, 6), "Ke shape mismatch: expected (6,6)"

        dof_map = [0, 1, 5, 6, 7, 11]
        self.Ke = expand_dof_mapping(Ke_reduced, full_size=12, dof_map=dof_map)

    def _integrand_stiffness_matrix(self, shape_data, xi: float) -> np.ndarray:
        _, dN_dxi, _ = shape_data[xi]
        B = self.strain_displacement_matrix(dN_dxi)
        return B.T @ self.material_stiffness_matrix() @ B

    def element_force_vector(self):
        """
        Computes the element force vector via numerical integration (Gauss quadrature).

        Returns:
        - np.ndarray: The force vector (12x1) in the global coordinate system.
        """
        gauss_points, weights = get_gauss_points(n=3, dim=1)
        xi_array = np.array([xi[0] for xi in gauss_points]).reshape(-1, 1)
        element_index = self.get_element_index()

        # Compute physical coordinates
        x_phys_array = self.jacobian_matrix[0, 0] * xi_array + self.mesh_dictionary["node_coordinates"][self.mesh_dictionary["connectivity"][element_index]].mean(axis=0)
        
        shape_data = {xi[0]: self.shape_functions(xi[0]) for xi in gauss_points}
        q_xi_array = np.array([interpolate_loads(x, self.load_array) for x in x_phys_array])

        Fe_reduced = sum(
            weights[i] * self._integrand_force_vector(shape_data, q_xi_array, xi_array[i, 0]) * self.detJ
            for i in range(len(weights))
        )

        assert Fe_reduced.shape == (6,), "Fe shape mismatch: expected (6,)"

        dof_map = [0, 1, 5, 6, 7, 11]
        self.Fe = expand_dof_mapping(Fe_reduced, full_size=12, dof_map=dof_map)

    def _integrand_force_vector(self, shape_data, q_xi_array, xi: float) -> np.ndarray:
        N, _, _ = shape_data[xi]
        print(q_xi_array.shape)
        return N.T @ q_xi_array

    def validate_matrices(self):
        """Validates that the stiffness and force matrices have the correct dimensions."""
        assert self.Ke.shape == (12, 12), f"Ke shape mismatch: Expected (12,12), got {self.Ke.shape}"
        assert self.Fe.shape == (12,), f"Fe shape mismatch: Expected (12,), got {self.Fe.shape}"
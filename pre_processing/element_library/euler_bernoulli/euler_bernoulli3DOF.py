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
        Debug prints added to verify tensor shapes before operations.

        Returns:
        - np.ndarray: The element stiffness matrix (12x12).
        """
        # Get Gauss points & weights (3-point Gauss quadrature)
        gauss_points, weights = get_gauss_points(n=3, dim=1)

        # Precompute shape functions for each Gauss point
        shape_data = {xi[0]: self.shape_functions(xi[0]) for xi in gauss_points}

        # Material stiffness matrix
        D = self.material_stiffness_matrix()

        # Debug print for material stiffness matrix
        print("\n===== Material Stiffness Matrix =====")
        pprint.pprint(f"D =\n{D}")
        print(f"D.shape = {D.shape} (Expected: (2,2) or material-specific)\n")

        # Define integrand function inside element_stiffness_matrix
        def integrand_stiffness_matrix(xi: float) -> np.ndarray:
            """
            Computes the stiffness integrand at a Gauss point.
        
            Parameters:
            - xi (float): Gauss point in natural coordinates.

            Returns:
            - np.ndarray: The stiffness matrix contribution at xi.
            """
            _, dN_dxi, _ = shape_data[xi]  # Extract shape function derivatives
            B = self.strain_displacement_matrix(dN_dxi)

            # Debug print for B matrix
            print("\n===== Strain-Displacement Matrix (B) =====")
            pprint.pprint(f"B =\n{B}")
            print(f"B.shape = {B.shape} (Expected: (6,6))\n")

            return B.T @ D @ B  # Matrix multiplication

        # Compute stiffness matrix via numerical integration
        Ke_reduced = sum(
            weights[i] * integrand_stiffness_matrix(gauss_points[i][0]) * self.detJ
            for i in range(len(weights))
        )

        # Debug print for reduced stiffness matrix
        print("\n===== Reduced Stiffness Matrix (After Integration) =====")
        pprint.pprint(f"Ke_reduced =\n{Ke_reduced}")
        print(f"Ke_reduced.shape = {Ke_reduced.shape} (Expected: (6,6))\n")

        assert Ke_reduced.shape == (6, 6), f"Ke_reduced shape mismatch: expected (6,6), got {Ke_reduced.shape}"

        # Map to 12 DOFs using provided mapping
        dof_map = [0, 1, 5, 6, 7, 11]
        self.Ke = expand_dof_mapping(Ke_reduced, full_size=12, dof_map=dof_map)

        # Debug print for final stiffness matrix after mapping
        print("\n===== Final Element Stiffness Matrix (Mapped to 12 DOFs) =====")
        pprint.pprint(f"self.Ke =\n{self.Ke}")
        print(f"self.Ke.shape = {self.Ke.shape} (Expected: (12,12))\n")


    def element_force_vector(self):
        """
        Computes the element force vector via fully vectorized numerical integration (Gauss quadrature).
        Debug prints added to verify tensor shapes before operations.

        Returns:
        - np.ndarray: The force vector (12x1) in the global coordinate system.
        """
        # Get Gauss points & weights
        gauss_points, weights = get_gauss_points(n=3, dim=1)  # 3-point Gauss quadrature
        xi_array = np.array([xi[0] for xi in gauss_points]).reshape(-1, 1)  # Shape: (3,1)
        element_index = self.get_element_index()

        # Compute physical coordinates for Gauss points
        x_phys_array = (
            self.jacobian_matrix[0, 0] * xi_array 
            + self.mesh_dictionary["node_coordinates"][self.mesh_dictionary["connectivity"][element_index]].mean(axis=0)
        )[:, 0]  # Extract only x-coordinates, shape (3,)

        # Compute shape function matrices for each Gauss point and stack them into a tensor
        shape_matrices = np.array([self.shape_functions(xi[0])[0] for xi in gauss_points])  # Shape: (3, 6, 1)

        # Debug print for shape functions
        print("\n===== Shape Function Matrix (Before Integration) =====")
        pprint.pprint(f"shape_matrices =\n{shape_matrices}")
        print(f"shape_matrices.shape = {shape_matrices.shape} (Expected: (3,6,1))\n")

        # Vectorized interpolation of loads at Gauss points (Ensuring shape (3, 6))
        q_xi_array = interpolate_loads(x_phys_array, self.load_array)  # Expected shape: (3,6)

        # Ensure q_xi_array has the right shape, even if x_phys_array is a single value
        if q_xi_array.ndim == 1:
            q_xi_array = q_xi_array.reshape(1, -1)  # Convert (6,) → (1,6)

        assert q_xi_array.shape == (3, 6), f"q_xi_array shape mismatch: expected (3,6), got {q_xi_array.shape}"

        # Debug print for interpolated force tensor
        print("\n===== Interpolated Load Vector (Before Integration) =====")
        pprint.pprint(f"q_xi_array =\n{q_xi_array}")
        print(f"q_xi_array.shape = {q_xi_array.shape} (Expected: (3,6))\n")

        # **Fully vectorized integration step** using einsum (performs Σ Wi * N.T * q_xi)
        Fe_reduced = np.einsum("i,ijk,ik->j", weights, shape_matrices, q_xi_array) * self.detJ

        # Debug print for reduced force vector
        print("\n===== Reduced Force Vector (After Integration) =====")
        pprint.pprint(f"Fe_reduced = {Fe_reduced}")
        print(f"Fe_reduced.shape = {Fe_reduced.shape} (Expected: (6,))\n")

        assert Fe_reduced.shape == (6,), f"Fe_reduced shape mismatch: expected (6,), got {Fe_reduced.shape}"

        # Map to 12 DOFs using provided mapping
        dof_map = [0, 1, 5, 6, 7, 11]
        self.Fe = expand_dof_mapping(Fe_reduced, full_size=12, dof_map=dof_map)

        # Debug print for final force vector after mapping
        print("\n===== Final Element Force Vector (Mapped to 12 DOFs) =====")
        pprint.pprint(f"self.Fe =\n{self.Fe}")
        print(f"self.Fe.shape = {self.Fe.shape} (Expected: (12,))\n")


    def validate_matrices(self):
        """Validates that the stiffness and force matrices have the correct dimensions."""
        assert self.Ke.shape == (12, 12), f"Ke shape mismatch: Expected (12,12), got {self.Ke.shape}"
        assert self.Fe.shape == (12,), f"Fe shape mismatch: Expected (12,), got {self.Fe.shape}"
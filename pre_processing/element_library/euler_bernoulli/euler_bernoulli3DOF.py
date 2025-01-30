import numpy as np
import pprint
import logging
from pre_processing.element_library.element_1D_base import Element1DBase
from pre_processing.element_library.utilities.dof_mapping import expand_stiffness_matrix, expand_force_vector
from pre_processing.element_library.utilities.shape_function_library.euler_bernoulli_sf import euler_bernoulli_shape_functions
from pre_processing.element_library.utilities.gauss_quadrature import get_gauss_points
from pre_processing.element_library.utilities.interpolate_loads import interpolate_loads

# Configure the logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the desired logging level

# You can add handlers to the logger if not already configured elsewhere
# For example, to log to a file:
# handler = logging.FileHandler('element_debug.log')
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

class EulerBernoulliBeamElement3DOF(Element1DBase):
    """
    1D structural member governed by Euler-Bernoulli beam theory modelling explciitly axial u_x, bending u_y effects and implicitly rotation θ_z through bending curvature.
    (u_x, u_y, 0, 0, 0, θ_z).
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
    
        # Extract the connectivity and element_length for the current element
        connectivity = self.mesh_dictionary["connectivity"][element_index]
        element_length = self.mesh_dictionary["element_lengths"][element_index]
    
        # Get the x-coordinates of the nodes in the current element
        x_node_coordinate = self.mesh_dictionary["node_coordinates"][connectivity][:, 0]  # Extract x-coordinates
    
        # DOUBLE CHECK that node_coordinates actually aligns with connectivity (which now references indices directly).
    
        # Compute the shape function derivatives in natural coordinates (at ξ = 0)
        _, dN_dxi, _ = euler_bernoulli_shape_functions(0.0, element_length)
    
        # Extract only the first and fourth shape function derivatives (indices 0 and 3)
        dN_dxi_specific = np.array([dN_dxi[0], dN_dxi[3]])  # Shape: (2,)
    
        # Compute the Jacobian matrix (dx/dxi) by multiplying dN_dxi with node coordinates
        jacobian_matrix = np.dot(dN_dxi_specific, x_node_coordinate)  # Shape: (1, 1) for 1D elements
    
        # Ensure jacobian_matrix is treated correctly (reshape to 2D if needed)
        jacobian_matrix = np.array([[jacobian_matrix]])  # Explicitly reshape to (1,1) for consistency
    
        # Compute the determinant of the Jacobian matrix (absolute value to ensure positivity)
        jacobian_determinant = np.abs(jacobian_matrix[0, 0])
    
        jacobian_verification = element_length / jacobian_determinant

        # Log the Jacobian computation verification
        logger.debug("===== Jacobian Computation Verification =====")
        jacobian_info = {
            "Jacobian Matrix": jacobian_matrix,
            "Jacobian Determinant": jacobian_determinant,
            "Element Length": element_length,
            "Element Length / Jacobian Determinant": jacobian_verification,
            "Expected Theoretical Value (2)": 2.0
        }
        logger.debug(pprint.pformat(jacobian_info))
        logger.debug("============================================")
    
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
        B_axial[3] = dN_dxi[1] * dxi_dx  # Axial strain contribution

        B_bending = np.zeros(6)
        B_bending[1] = dN_dxi[2] * (dxi_dx ** 2)  # Bending curvature
        B_bending[4] = dN_dxi[3] * (dxi_dx ** 2)  # Bending curvature

        # Assemble strain-displacement matrix
        return np.vstack([B_axial, B_bending])  # Shape: (2,6)
    
    def element_stiffness_matrix(self):
        """Computes the element stiffness matrix using vectorized Gauss quadrature integration."""
        gauss_points, weights = get_gauss_points(n=3, dim=1)
        shape_data = {xi[0]: self.shape_functions(xi[0]) for xi in gauss_points}
        D = self.material_stiffness_matrix()

        logger.debug("===== Material Stiffness Matrix =====")
        logger.debug("D =\n%s", D)
        logger.debug("D.shape = %s (Expected: (2,2) or material-specific)", D.shape)
        logger.debug("=====================================")

        def integrand_stiffness_matrix(xi: float) -> np.ndarray:
            """Computes the stiffness integrand at a Gauss point."""
            _, dN_dxi, _ = shape_data[xi]
            B = self.strain_displacement_matrix(dN_dxi)

            logger.debug("===== Strain-Displacement Matrix (B) =====")
            logger.debug("B =\n%s", B)
            logger.debug("B.shape = %s (Expected: (2,6))", B.shape)
            logger.debug("==========================================")

            return B.T @ D @ B  

        # **Vectorized integration using einsum** (performs Σ Wi * B.T * D * B)
        Ke_reduced = np.einsum(
            "i,ijk->jk", weights, np.array([integrand_stiffness_matrix(xi[0]) for xi in gauss_points])
        ) * self.detJ

        logger.debug("===== Reduced Stiffness Matrix (After Integration) =====")
        logger.debug("Ke_reduced =\n%s", Ke_reduced)
        logger.debug("Ke_reduced.shape = %s (Expected: (6,6))", Ke_reduced.shape)
        logger.debug("=======================================================")

        self.Ke = expand_stiffness_matrix(Ke_reduced, full_size=12, dof_map=self.dof_map)

        logger.debug("===== Final Element Stiffness Matrix (Mapped to 12 DOFs) =====")
        logger.debug("self.Ke =\n%s", self.Ke)
        logger.debug("self.Ke.shape = %s (Expected: (12,12))", self.Ke.shape)
        logger.debug("=============================================================")

    def element_force_vector(self):
        """
        Computes the element force vector using vectorized numerical integration (Gauss quadrature).
        Detailed debug logs are added to verify tensor shapes before operations.

        Returns:
        - np.ndarray: The force vector (12x1) in the global coordinate system.
        """
        # Get Gauss points & weights (3-point Gauss quadrature)
        gauss_points, weights = get_gauss_points(n=3, dim=1)
        xi_array = np.array([xi[0] for xi in gauss_points]).reshape(-1, 1)  # Shape: (3,1)
        element_index = self.get_element_index()

        # Compute physical coordinates for Gauss points
        x_phys_array = (
            self.jacobian_matrix[0, 0] * xi_array 
            + self.mesh_dictionary["node_coordinates"][self.mesh_dictionary["connectivity"][element_index]].mean(axis=0)
        )[:, 0]  # Extract only x-coordinates, shape (3,)

        # Compute shape function matrices for each Gauss point and stack them into a tensor
        shape_matrices = np.stack(
            [self.shape_functions(xi[0])[0] for xi in gauss_points], axis=0
        )[:, np.newaxis, :]  # Shape: (3, 1, 6)

        # Log shape function matrix
        logger.debug("===== Shape Function Matrix (Before Integration) =====")
        logger.debug("shape_matrices =\n%s", shape_matrices)
        logger.debug("shape_matrices.shape = %s (Expected: (3,1,6))", shape_matrices.shape)
        logger.debug("=======================================================")

        # Vectorized interpolation of loads at Gauss points (Ensuring shape (3, 6))
        q_xi_array = interpolate_loads(x_phys_array, self.load_array)  # Expected shape: (3,6)

        # Ensure q_xi_array has the right shape, even if x_phys_array is a single value
        if q_xi_array.ndim == 1:
            q_xi_array = q_xi_array.reshape(1, -1)  # Convert (6,) → (1,6)

        assert q_xi_array.shape == (3, 6), f"q_xi_array shape mismatch: expected (3,6), got {q_xi_array.shape}"

        # Log interpolated load tensor
        logger.debug("===== Interpolated Load Vector (Before Integration) =====")
        logger.debug("q_xi_array =\n%s", q_xi_array)
        logger.debug("q_xi_array.shape = %s (Expected: (3,6))", q_xi_array.shape)
        logger.debug("===========================================================")

        # Log einsum constituent tensor shapes
        logger.debug("===== Einsum Constituent Tensor Shapes (Before Integration) =====")
        logger.debug("weights.shape = %s", weights.shape)
        logger.debug("shape_matrices.shape = %s", shape_matrices.shape)
        logger.debug("q_xi_array.shape = %s", q_xi_array.shape)
        logger.debug("=====================================================================")

        def integrand_force_vector(xi_idx: int) -> np.ndarray:
            """
            Computes the force integrand at a given Gauss point index.

            Parameters:
            - xi_idx (int): Index of the Gauss point.

            Returns:
            - np.ndarray: Force contribution at xi_idx.
            """
            N = self.shape_functions(gauss_points[xi_idx][0])[0]
            return N.T @ q_xi_array[xi_idx]  # Shape: (6,)

        # **Fully vectorized integration step** using einsum (performs Σ Wi * N.T * q_xi)
        Fe_reduced = np.einsum(
            "i,ik->k", weights, np.array([integrand_force_vector(i) for i in range(len(weights))])
        ) * self.detJ  # Shape: (6,)

        # Log reduced force vector (before mapping)
        logger.debug("===== Reduced Force Vector (After Integration) =====")
        logger.debug("Fe_reduced = %s", Fe_reduced)
        logger.debug("Fe_reduced.shape = %s (Expected: (6,))", Fe_reduced.shape)
        logger.debug("=====================================================")

        assert Fe_reduced.shape == (6,), f"Fe_reduced shape mismatch: expected (6,), got {Fe_reduced.shape}"

        # Map to 12 DOFs using provided mapping
        self.Fe = expand_force_vector(Fe_reduced, full_size=12, dof_map=self.dof_map)

    # Log final mapped force vector
    logger.debug("===== Final Element Force Vector (Mapped to 12 DOFs) =====")
    logger.debug("self.Fe =\n%s", self.Fe)
    logger.debug("self.Fe.shape = %s (Expected: (12,))", self.Fe.shape)
    logger.debug("==========================================================")

    def validate_matrices(self):
        """Validates that the stiffness and force matrices have the correct dimensions."""
        assert self.Ke.shape == (12, 12), f"Ke shape mismatch: Expected (12,12), got {self.Ke.shape}"
        assert self.Fe.shape == (12,), f"Fe shape mismatch: Expected (12,), got {self.Fe.shape}"
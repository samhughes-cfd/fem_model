# pre_processing\element_library\euler_bernoulli\euler_bernoulli3DOF.py

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

        self.dof_map = self.get_dof_mapping()  # Get the DOF mapping for this element

    def get_dof_mapping(self):
        """
        Returns DOF mapping for this element, (u_x, u_y, 0, 0, 0, θ_z).
        
        Node 1: u_x (0), u_y (1), 0, 0, 0, θ_z (5)
        Node 2: u_x (6), u_y (7), 0, 0, 0, θ_z (11)
        
        """
        return [0, 1, 5, 6, 7, 11]  

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
        # Assemble mayerial stiffness matrix
        return np.diag([self.E * self.A, self.E * self.I_z]) # Shape: (2,2)
        
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
    
        # Get Gauss points and weights
        gauss_points, weights = get_gauss_points(n=3, dim=1)

        # **Vectorized computation of shape_tensor N, dN_dxi, d2N_dxi2**
        # Since shape_tensor().shape (2,6),
        # Assemble B_tensor (n,2,6)
        shape_tensor = np.array([
            self.shape_functions(xi[0]) for xi in gauss_points
        ])
    
        # Get the material stiffness matrix D (should be (2,2))
        D = self.material_stiffness_matrix()

        # **Vectorized computation of B tensor**
        # Since strain_displacement_matrix().shape (2,6),
        # Assemble B_tensor (n,2,6)
        B_tensor = np.array([
            self.strain_displacement_matrix(shape_tensor[xi[0]][1])
            for xi in gauss_points
        ])

        # Log each B matrix for every Gauss point (printed horizontally)
        for idx, (xi, B_tensor) in enumerate(zip(gauss_points, B_tensor)):
            logger.debug("===== B Matrix for Gauss Point %d =====", idx)
            logger.debug("Gauss Point: %s", xi)
            logger.debug("B matrix (horizontal):\n%s", np.array2string(B_tensor, separator=', '))
            logger.debug("B matrix shape: %s", B_tensor.shape)
            logger.debug("=====================================")

        # **Vectorized computation of B_T tensor**
        # Since strain_displacement_matrix().T.shape (6,2),
        # Assemble B_tensor_T (n,6,2)
        B_tensor_T = np.array([
            self.strain_displacement_matrix(shape_data[xi[0]][1]).T
            for xi in gauss_points
        ])

        # Log each transposed B matrix for every Gauss point (printed horizontally)
        for idx, (xi, B_tensor_T) in enumerate(zip(gauss_points, B_tensor_T)):
            logger.debug("===== Transposed B Matrix for Gauss Point %d =====", idx)
            logger.debug("Gauss Point: %s", xi)
            logger.debug("B matrix (transposed, horizontal):\n%s", np.array2string(B_tensor_T, separator=', '))
            logger.debug("B matrix (transposed) shape: %s", B_tensor_T.shape)
            logger.debug("=====================================")

        # Log and check the shapes of the tensors.
        logger.debug(f"weights.shape: {weights.shape}")
        # get_gauss_points(n=3, dim=1) calls the no. of gauss points "n" for no. of dimensions "dim". Using the extracted (gauss_points, weights) the assembled gauss weight vector is of shape (3,)
        assert weights.shape == (3,), f"Unexpected shape for weights: {weights.shape}"

        logger.debug(f"B_matrices.shape: {B_tensor.shape}")
        # Since strain_displacement_matrix() returns a B matrix of (2,6), B_matrices (looped over the gauss weights) should have shape (3,2,6)
        assert B_tensor.shape == (3, 2, 6), f"Unexpected shape for B_matrices: {B_tensor.shape}"

        logger.debug(f"D.shape: {D.shape}")
        # material_stiffness_matrix
        assert D.shape == (2, 2), f"Unexpected shape for D matrix: {D.shape}"

        logger.debug(f"B_matrices_T.shape: {B_tensor_T.shape}")
        # The transposed array should have shape (3,6,2)
        assert B_tensor_T.shape == (3, 6, 2), f"Unexpected transposed shape for B_matrices_T: {B_tensor_T.shape}"

        # **Vectorized integration using einsum (Σ Wi * B.T * D * B)**
        try:
            Ke_reduced = (
                np.einsum("i,imk,mn,inj->kj", weights, B_tensor_T, D, B_tensor)
                * self.detJ
            )
            logging.info("✅ Stiffness matrix computed successfully. Shape: %s", Ke_reduced.shape)
        except ValueError as e:
            logging.error("Einsum operation failed: %s", e)
            logging.error("weights.shape: %s", weights.shape)
            logging.error("B_matrices_T.shape: %s", B_tensor_T.shape)
            logging.error("D.shape: %s", D.shape)
            logging.error("B_matrices.shape: %s", B_tensor.shape)
            raise  # Re-raise the error for traceback

        logger.debug("===== Reduced Stiffness Matrix (After Integration) =====")
        logger.debug("Ke_reduced =\n%s", np.array2string(Ke_reduced, separator=', '))
        logger.debug("Ke_reduced.shape = %s (Expected: (6,6))", Ke_reduced.shape)
        logger.debug("=======================================================")

        # Expand the reduced stiffness matrix to full DOF size (12x12)
        self.Ke = expand_stiffness_matrix(Ke_reduced, full_size=12, dof_map=self.dof_map)

        logger.debug("===== Final Element Stiffness Matrix (Mapped to 12 DOFs) =====")
        logger.debug("self.Ke =\n%s", np.array2string(self.Ke, separator=', '))
        logger.debug("self.Ke.shape = %s (Expected: (12,12))", self.Ke.shape)
        logger.debug("=============================================================")

    def element_force_vector(self):
        """
        Computes the element force vector using vectorized numerical integration (Gauss quadrature).

        Returns:
        - np.ndarray: The force vector (12x1) in the global coordinate system.
        """
        # Step 1: Get Gauss points & weights (3-point Gauss quadrature)
        gauss_points, weights = get_gauss_points(n=3, dim=1)
        xi_array = np.array([xi[0] for xi in gauss_points]).reshape(-1, 1)  # Shape: (3,1)
        element_index = self.get_element_index()

        # Step 2: Compute physical coordinates for Gauss points
        x_phys_array = (
            self.jacobian_matrix[0, 0] * xi_array 
            + self.mesh_dictionary["node_coordinates"][self.mesh_dictionary["connectivity"][element_index]].mean(axis=0)
        )[:, 0]  # Extract only x-coordinates, shape (3,)

        # Step 3: Compute shape function matrices for each Gauss point
        shape_matrices = np.array([
            self.shape_functions(xi[0])[0].reshape(6, 1)  # Ensure (6,1) shape per Gauss point
            for xi in gauss_points
        ])  # Shape: (3,6,1)

        # Log shape function matrix
        #logger.debug("===== Shape Function Matrix (Before Integration) =====")
        #logger.debug("shape_matrices =\n%s", shape_matrices)
        #logger.debug("shape_matrices.shape = %s (Expected: (3,1,6))", shape_matrices.shape)
        #logger.debug("=======================================================")


        # Step 4: Interpolate loads at Gauss points
        q_xi_array = interpolate_loads(x_phys_array, self.load_array)  # Expected shape: (3,6)

        # Ensure q_xi_array has the correct shape, even if x_phys_array is a single value
        if q_xi_array.ndim == 1:
            q_xi_array = q_xi_array.reshape(1, -1)  # Convert (6,) → (1,6)

        assert q_xi_array.shape == (3, 6), f"q_xi_array shape mismatch: expected (3,6), got {q_xi_array.shape}"

        # Log interpolated load tensor
        #logger.debug("===== Interpolated Load Vector (Before Integration) =====")
        #logger.debug("q_xi_array =\n%s", q_xi_array)
        #logger.debug("q_xi_array.shape = %s (Expected: (3,6))", q_xi_array.shape)
        #logger.debug("===========================================================")

        # Log einsum constituent tensor shapes
        #logger.debug("===== Einsum Constituent Tensor Shapes (Before Integration) =====")
        #logger.debug("weights.shape = %s", weights.shape)
        #logger.debug("shape_matrices.shape = %s", shape_matrices.shape)
        #logger.debug("q_xi_array.shape = %s", q_xi_array.shape)
        #logger.debug("=====================================================================")

        # Step 5: Fully vectorized integration using einsum
        Fe_reduced = np.einsum("i,ijk,ik->j", weights, shape_matrices, q_xi_array) * self.detJ  # Shape: (6,)

        # Log reduced force vector (before mapping)
        #logger.debug("===== Reduced Force Vector (After Integration) =====")
        #logger.debug("Fe_reduced = %s", Fe_reduced)
        #logger.debug("Fe_reduced.shape = %s (Expected: (6,))", Fe_reduced.shape)
        #logger.debug("=====================================================")

        # Step 6: Map to 12 DOFs using provided mapping
        self.Fe = expand_force_vector(Fe_reduced, full_size=12, dof_map=self.dof_map)

        # Log final mapped force vector
        #logger.debug("===== Final Element Force Vector (Mapped to 12 DOFs) =====")
        #logger.debug("self.Fe =\n%s", self.Fe)
        #logger.debug("self.Fe.shape = %s (Expected: (12,))", self.Fe.shape)
        #logger.debug("==========================================================")

    def validate_matrices(self):
        """Validates that the stiffness and force matrices have the correct dimensions."""
        assert self.Ke.shape == (12, 12), f"Ke shape mismatch: Expected (12,12), got {self.Ke.shape}"
        assert self.Fe.shape == (12,), f"Fe shape mismatch: Expected (12,), got {self.Fe.shape}"
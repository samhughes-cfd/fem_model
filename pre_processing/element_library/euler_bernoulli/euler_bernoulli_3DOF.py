# pre_processing\element_library\euler_bernoulli\euler_bernoulli_3DOF.py

import numpy as np
import pprint
import logging
from pre_processing.element_library.element_1D_base import Element1DBase
from pre_processing.element_library.utilities.dof_mapping import expand_stiffness_matrix, expand_force_vector
from pre_processing.element_library.euler_bernoulli.utilities.shape_functions_3DOF import euler_bernoulli_shape_functions
from pre_processing.element_library.euler_bernoulli.utilities.element_stiffness_matrix_3DOF import compute_stiffness_matrix
from pre_processing.element_library.euler_bernoulli.utilities.element_force_vector_3DOF import compute_force_vector

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

        self.dof_map_binary= self.get_dof_map_binary()  # Get the DOF mapping for this element

    def get_dof_map_binary(self):
        """
        Returns the binary DOF mapping and active indices for the EulerBernoulliBeamElement3DOF element.
    
        Full DOF size: 12 (2 nodes x 6 DOF per node):
            - Node 1: u_x (0), u_y (1), u_z (2), θ_x (3), θ_y (4), θ_z (5)
            - Node 2: u_x (6), u_y (7), u_z (8), θ_x (9), θ_y (10), θ_z (11)

        EulerBernoulliBeamElement3DOF element uses 6 DOFs (2 nodes x 3 DOF per node):
            - Node 1: u_x (0), u_y (1), 0 (2), 0 (3), 0 (4), θ_z (5)
            - Node 2: u_x (6), u_y (7), 0 (8), 0 (9), 0 (10), θ_z (11)

        Binary DOF mapping for active indices in this element:
            - [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]
            - A 1 indicates an active DOF and 0 indicates an inactive DOF.
            - Active DOFs:
                - Node 1: u_x, u_y, θ_z 
                - Node 2: u_x, u_y, θ_z 

        Returns:
            tuple: A tuple containing:
                - A list of 12 integers (binary mapping).
                - A NumPy array of the active DOF indices.
        """
        return [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1] 



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

        # Compute the shape function derivatives in natural coordinates (at ξ = 0)
        _, dN_dxi_matrix, _ = self.shape_functions(0.0)  # Shape: (1, 2, 6)

        # Extract only the first and fourth axial shape function derivatives
        dN_dxi_specific = dN_dxi_matrix[0, 0, [0, 3]]  # Select row 0 (axial strain) and cols 0,3

        # Compute the Jacobian matrix (dx/dxi)
        jacobian_matrix = np.dot(dN_dxi_specific, x_node_coordinate)  # Shape: (1, 1) for 1D elements

        # Ensure jacobian_matrix is treated correctly (reshape to 2D if needed)
        jacobian_matrix = np.array([[jacobian_matrix]])  # Explicitly reshape to (1,1) for consistency

        # Compute the determinant of the Jacobian matrix
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

    
    def shape_functions(self, xi) -> tuple:
        """
        Evaluates the 2×6 shape function matrices and their derivatives at the given evaluation points.
    
        Parameters
        ----------
        xi : float or ndarray
            The natural coordinate(s), in the range [-1, 1]. Can be a scalar or a 1D array.
    
        Returns
        -------
        tuple of ndarray
            A tuple (N_matrix, dN_dxi_matrix, d2N_dxi2_matrix) where each array has shape (n, 2, 6),
            with n being the number of evaluation points (e.g., Gauss points). These matrices are used both
            for assembling the element stiffness matrix (via numerical integration) and for interpolating the 
            continuous displacement field once the global system is solved.
        """
        element_index = self.get_element_index()
        element_length = self.mesh_dictionary["element_lengths"][element_index]
        return euler_bernoulli_shape_functions(xi, element_length, poly_order=3)

    def material_stiffness_matrix(self) -> np.ndarray:
        """
        Constructs the element material stiffness matrix.

        Returns:
        - np.ndarray: Diagonal matrix representing axial and bending stiffness.
        """
        # Assemble mayerial stiffness matrix
        return np.diag([self.E * self.A, self.E * self.I_z]) # Shape: (2,2)
        
    def strain_displacement_matrix(self, dN_dxi_matrix: np.ndarray) -> np.ndarray:
        """
        Computes the strain-displacement matrix B in physical coordinates for an Euler–Bernoulli beam element.
    
        This function transforms the natural coordinate derivative matrix (dN/dξ) into physical coordinates (dN/dx),
        making it suitable for computing strain, stress, internal forces, and energy quantities.

        Parameters
        ----------
        dN_dxi_matrix : ndarray
            Shape function derivative matrix in natural coordinates:
            - (2,6) for a single evaluation point.
            - (n,2,6) for `n` evaluation points.
            - Convention:
                - Row 0: [ dN1/dxi,  0,       0,  dN4/dxi,   0,       0  ]
                - Row 1: [    0,  dN2/dxi,  dN3/dxi,   0,  dN5/dxi,  dN6/dxi ]

        Returns
        -------
        B : ndarray
            Strain-displacement matrix in physical coordinates:
            - (2,6) for a single evaluation point.
            - (n,2,6) for `n` evaluation points.
            - Convention:
                - Row 0: [ dN1/dx,  0,       0,  dN4/dx,   0,       0  ]
                - Row 1: [    0,  dN2/dx,  dN3/dx,   0,  dN5/dx,  dN6/dx ]
            - Scaling:
                - **Axial strain** components scale by **(1/detJ)**.
                - **Bending strain** components scale by **(1/detJ)²**.

        Notes
        -----
        - The Jacobian transformation converts derivatives from the natural coordinate system (ξ) to the 
        physical coordinate system (x), ensuring correct strain recovery.
        - This function is **only used for post-processing** after solving `F = K U`, since Ke is assembled 
        directly using `dN_dxi_matrix` in natural coordinates.
        """

        # Compute transformation factor (Jacobian inverse)
        if np.abs(self.detJ) < 1e-12:  # Avoid division errors for near-zero Jacobian
            raise ValueError("Jacobian determinant is too small, possible singular transformation.")

        dxi_dx = 1.0 / self.detJ

        # Ensure input is at least 3D (n,2,6)
        dN_dxi_matrix = np.atleast_3d(dN_dxi_matrix)

        # Allocate memory for the transformed matrix
        B = np.empty_like(dN_dxi_matrix)

        # Apply the transformation
        B[:, 0, :] = dN_dxi_matrix[:, 0, :] * dxi_dx         # Axial component: (1/detJ) * dN/dξ
        B[:, 1, :] = dN_dxi_matrix[:, 1, :] * (dxi_dx ** 2)  # Bending component: (1/detJ)² * dN/dξ

        return B.squeeze()  # Ensures consistent output: (2,6) for a single point, (n,2,6) for multiple points

    def element_stiffness_matrix(self):
        """
        Computes the element stiffness matrix (12x12) using Gauss quadrature.
        Expands the reduced 6x6 matrix to full DOF size.
        """
        Ke_reduced = compute_stiffness_matrix(self)  # Get 6x6 reduced matrix
        self.Ke = expand_stiffness_matrix(Ke_reduced, full_size=12, dof_map_binary=self.dof_map_binary)  # Expand to 12x12
        return self.Ke

    def element_force_vector(self):
        """
        Computes the element force vector (12x1) using Gauss quadrature.
        Expands the reduced 6x1 vector to full DOF size.
        """
        Fe_reduced = compute_force_vector(self)  # Get 6x1 reduced vector
        self.Fe = expand_force_vector(Fe_reduced, full_size=12, dof_map_binary=self.dof_map_binary)  # Expand to 12x1
        return self.Fe


    def validate_matrices(self):
        """Validates that the stiffness and force matrices have the correct dimensions."""
        assert self.Ke.shape == (12, 12), f"Ke shape mismatch: Expected (12,12), got {self.Ke.shape}"
        assert self.Fe.shape == (12,), f"Fe shape mismatch: Expected (12,), got {self.Fe.shape}"
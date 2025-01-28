import numpy as np
import logging
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from pre_processing.element_library.element_registry import get_euler_bernoulli, get_timoshenko

from pre_processing.element_library.euler_bernoulli.euler_bernoulli import EulerBernoulliBeamElement
from pre_processing.element_library.timoshenko import TimoshenkoBeamElement
from pre_processing.element_library.utilities.jacobian import compute_jacobian_matrix, compute_jacobian_determinant


class Element1DBase:
    """
    Base class for 1D finite elements. Provides core functionalities:
    - Stores geometry, material, and mesh data.
    - Instantiates elements in **batch** for computational efficiency.
    - Uses parallelization and vectorized computations for efficiency.
    - Precomputes Jacobians to speed up element matrix calculations.
    """

    # Element type mapping dictionary
    ELEMENT_TYPE_MAP = {
        "EulerBernoulliBeamElement": EulerBernoulliBeamElement,
        "TimoshenkoBeamElement": TimoshenkoBeamElement,
    }

    def __init__(self, geometry_array, material_array, mesh_dictionary, load_array, dof_per_node=6):
        """
        Initializes the base 1D finite element system.

        Args:
            geometry_array (np.ndarray): Geometry properties (1, 20).
            material_array (np.ndarray): Material properties (1, 4).
            mesh_dictionary (dict): Mesh data (node IDs, connectivity, etc.).
            load_array (np.ndarray): External loads (N, 9).
            dof_per_node (int, optional): DOFs per node. Default is 6.
        """
        self.geometry_array = geometry_array
        self.material_array = material_array
        self.mesh_dictionary = mesh_dictionary
        self.load_array = load_array
        self.dof_per_node = dof_per_node  # Default max DOFs per node = 6

        # Extract mesh information
        self._element_types = mesh_dictionary.get("element_types", np.array([], dtype=str))
        self._element_ids = mesh_dictionary.get("element_ids_array", np.array([], dtype=int))
        self._connectivity = mesh_dictionary.get("connectivity", np.array([], dtype=int))
        self._node_coordinates = mesh_dictionary.get("node_coordinates", np.array([], dtype=float))
        self._element_lengths = mesh_dictionary.get("element_lengths_array", np.array([], dtype=float))  

        # Precompute Jacobians
        self._jacobians = self._precompute_jacobians()

        # Instantiate elements in parallel
        self.elements_instances = self._instantiate_elements()

        # Compute element stiffness and force matrices using NumPy broadcasting
        self.element_stiffness_matrices = self._compute_stiffness_matrices_vectorized()
        self.element_force_vectors = self._compute_force_vectors_vectorized()

        # Convert to sparse format for large systems
        self.element_stiffness_matrices = self._convert_to_sparse(self.element_stiffness_matrices)
        self.element_force_vectors = self._convert_to_sparse(self.element_force_vectors)

    def _precompute_jacobians(self):
        """
        Precomputes the Jacobian matrices and determinants for all elements.

        Returns:
            dict: {element_id: {"jacobian_matrix": J, "jacobian_determinant": detJ}}
        """
        jacobians = {}

        for idx, element_id in enumerate(self._element_ids):
            node_ids = self._connectivity[idx]
            node_coords = self._node_coordinates[node_ids - 1]  # Convert node IDs to 0-based index

            # Compute Jacobian matrix and determinant
            J = compute_jacobian_matrix(node_coords.reshape(-1, 1))
            detJ = compute_jacobian_determinant(J)

            if detJ <= 0:
                raise ValueError(f"Invalid Jacobian determinant ({detJ}) for Element ID {element_id}. Check node ordering.")

            jacobians[element_id] = {"jacobian_matrix": J, "jacobian_determinant": detJ}

        return jacobians

    def _instantiate_elements(self):
        """
        Instantiates all elements in parallel using joblib.

        Returns:
            np.ndarray: Array of instantiated element objects.
        """
        num_elements = len(self._connectivity)
        elements = Parallel(n_jobs=-1)(
            delayed(self._instantiate_single_element)(idx) for idx in range(num_elements)
        )
        return np.array(elements, dtype=object)

    def _instantiate_single_element(self, idx):
        """
        Instantiates a single element.

        Args:
            idx (int): Element index.

        Returns:
            object: Instantiated element object.
        """
        try:
            element_type = self._element_types[idx]
            element_class = self.ELEMENT_TYPE_MAP.get(element_type)

            if element_class is None:
                raise ValueError(f"Unsupported element type: {element_type}")

            return element_class(
                element_id=self._element_ids[idx],  
                material_array=self.material_array,
                geometry_array=self.geometry_array,
                mesh_dictionary=self.mesh_dictionary,
                loads_array=self.loads_array 
            )
        except Exception as e:
            logging.error(f"Error instantiating element {self._element_ids[idx]}: {e}", exc_info=True)
            return None

    def _compute_stiffness_matrices_vectorized(self):
        """
        Computes all stiffness matrices Ke using NumPy broadcasting.

        Returns:
            np.ndarray: Stiffness matrices of all elements.
        """
        return np.array([element.element_stiffness_matrix() for element in self.elements_instances])

    def _compute_force_vectors_vectorized(self):
        """
        Computes all force vectors Fe using NumPy broadcasting.

        Returns:
            np.ndarray: Force vectors of all elements.
        """
        return np.array([element.element_force_vector() for element in self.elements_instances])

    def _convert_to_sparse(self, matrix_array):
        """
        Converts an array of matrices to a list of sparse matrices.

        Args:
            matrix_array (np.ndarray): Input dense matrix array.

        Returns:
            list: List of sparse matrices.
        """
        return [csr_matrix(matrix) for matrix in matrix_array]

    def assemble_global_dof_indices(self, element_id):
        """
        Constructs the global DOF indices for an element.

        Args:
            element_id (int): The ID of the element.

        Returns:
            list: Global DOF indices.
        """
        element_index = np.where(self._element_ids == element_id)[0][0]  # Find element index
        node_ids = self._connectivity[element_index]  # Shape: (2,)
        global_dof_indices = []

        for node_id in node_ids:
            start_dof = (node_id - 1) * self.dof_per_node
            dof_indices = list(range(start_dof, start_dof + self.dof_per_node))
            global_dof_indices.extend(dof_indices)

        return global_dof_indices  # Length: num_nodes_per_element * dof_per_node

    def validate_matrices(self):
        """
        Ensures all Ke and Fe matrices have the correct dimensions.

        Raises:
            AssertionError: If any Ke or Fe matrix has incorrect dimensions.
        """
        expected_Ke_shape = (self.dof_per_node * 2, self.dof_per_node * 2)
        expected_Fe_shape = (self.dof_per_node * 2,)

        for idx, element in enumerate(self.elements_instances):
            if element is None:
                continue  # Skip if instantiation failed

            assert element.Ke.shape == expected_Ke_shape, (
                f"Element {self._element_ids[idx]}: Ke shape mismatch. "
                f"Expected {expected_Ke_shape}, got {element.Ke.shape}"
            )
            assert element.Fe.shape == expected_Fe_shape, (
                f"Element {self._element_ids[idx]}: Fe shape mismatch. "
                f"Expected {expected_Fe_shape}, got {element.Fe.shape}"
            )
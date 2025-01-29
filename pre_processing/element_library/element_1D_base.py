import numpy as np
from scipy.sparse import csr_matrix
from pre_processing.element_library.element_factory import create_elements_batch


class Element1DBase:
    """
    Base class for 1D finite elements. 

    Responsibilities:
    - Stores geometry, material, and mesh data.
    - Requests element instantiation from `element_factory.py`.
    - Computes element stiffness and force matrices.
    - Precomputes Jacobians to optimize matrix calculations.
    """

    def __init__(self, geometry_array, material_array, mesh_dictionary, load_array, dof_per_node=6):
        """
        Initializes the base 1D finite element system.

        Args:
            geometry_array (np.ndarray): Geometry properties.
            material_array (np.ndarray): Material properties.
            mesh_dictionary (dict): Mesh data including connectivity, element types, and node coordinates.
            load_array (np.ndarray): External loads applied to the system.
            dof_per_node (int, optional): Degrees of freedom per node (default: 6).
        """
        self.geometry_array = geometry_array
        self.material_array = material_array
        self.mesh_dictionary = mesh_dictionary
        self.load_array = load_array
        self.dof_per_node = dof_per_node  # Default max DOFs per node = 6

        # Request elements from `element_factory.py`
        self.elements_instances = self._instantiate_elements()

        # Compute element stiffness and force matrices
        self.element_stiffness_matrices = self._compute_stiffness_matrices_vectorized()
        self.element_force_vectors = self._compute_force_vectors_vectorized()

        # Convert to sparse format
        self.element_stiffness_matrices = self._convert_to_sparse(self.element_stiffness_matrices)
        self.element_force_vectors = self._convert_to_sparse(self.element_force_vectors)

    def _instantiate_elements(self):
        """
        Requests batch element instantiation from `element_factory.py`.

        Returns:
            np.ndarray: Array of instantiated element objects.
        """
        params_list = np.array([
            {
                "geometry_array": self.geometry_array,
                "material_array": self.material_array,
                "mesh_dictionary": self.mesh_dictionary,
                "load_array": self.load_array,
            }
            for _ in self.mesh_dictionary["element_ids"]
        ], dtype=object)

        return create_elements_batch(self.mesh_dictionary, params_list)

    def _compute_stiffness_matrices_vectorized(self):
        """
        Computes the element stiffness matrices (Ke) using NumPy broadcasting.

        Returns:
            np.ndarray: An array of element stiffness matrices.
        """
        return np.array([element.element_stiffness_matrix() for element in self.elements_instances])

    def _compute_force_vectors_vectorized(self):
        """
        Computes the element force vectors (Fe) using NumPy broadcasting.

        Returns:
            np.ndarray: An array of element force vectors.
        """
        return np.array([element.element_force_vector() for element in self.elements_instances])

    def _convert_to_sparse(self, matrix_array):
        """
        Converts an array of dense matrices to sparse format.

        Args:
            matrix_array (np.ndarray): Array of dense matrices.

        Returns:
            list: A list of sparse matrices in compressed sparse row (CSR) format.
        """
        return [csr_matrix(matrix) for matrix in matrix_array]

    def assemble_global_dof_indices(self, element_id):
        """
        Constructs the global DOF indices for an element.

        Args:
            element_id (int): The ID of the element.

        Returns:
            list: A list of global DOF indices associated with the element.
        """
        element_index = np.where(self.mesh_dictionary["element_ids"] == element_id)[0][0]  # ✅ FIXED
        node_ids = self.mesh_dictionary["connectivity"][element_index]  # ✅ FIXED
        global_dof_indices = []

        for node_id in node_ids:
            start_dof = (node_id - 1) * self.dof_per_node
            dof_indices = list(range(start_dof, start_dof + self.dof_per_node))
            global_dof_indices.extend(dof_indices)

        return global_dof_indices

    def validate_matrices(self):
        """
        Ensures all element stiffness matrices (Ke) and force vectors (Fe) have the correct dimensions.

        Raises:
            AssertionError: If any Ke or Fe matrix has incorrect dimensions.
        """
        expected_Ke_shape = (self.dof_per_node * 2, self.dof_per_node * 2)
        expected_Fe_shape = (self.dof_per_node * 2,)

        for idx, element in enumerate(self.elements_instances):
            if element is None:
                continue  # Skip if instantiation failed

            assert element.Ke.shape == expected_Ke_shape, (
                f"Element {self.mesh_dictionary['element_ids'][idx]}: Ke shape mismatch. "  # ✅ FIXED
                f"Expected {expected_Ke_shape}, got {element.Ke.shape}"
            )
            assert element.Fe.shape == expected_Fe_shape, (
                f"Element {self.mesh_dictionary['element_ids'][idx]}: Fe shape mismatch. "  # ✅ FIXED
                f"Expected {expected_Fe_shape}, got {element.Fe.shape}"
            )
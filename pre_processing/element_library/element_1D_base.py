# pre_processing\element_library\element_1D_base.py

import logging
import numpy as np
from scipy.sparse import coo_matrix
from pre_processing.element_library.element_factory import create_elements_batch

logger = logging.getLogger(__name__)


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
        logger.info("Initializing Element1DBase...")

        self.geometry_array = geometry_array
        self.material_array = material_array
        self.mesh_dictionary = mesh_dictionary
        self.load_array = load_array
        self.dof_per_node = dof_per_node  # Default max DOFs per node = 6

        # Request elements from `element_factory.py`
        logger.info("Instantiating elements from factory...")
        self.elements_instances = self._instantiate_elements()
        logger.info(f"✅ Successfully instantiated {len(self.elements_instances)} elements.")

        # Compute element stiffness and force matrices
        logger.info("Computing element stiffness matrices...")
        self.element_stiffness_matrices = self._compute_stiffness_matrices_vectorized()
        logger.info(f"✅ Computed {len(self.element_stiffness_matrices)} stiffness matrices.")

        logger.info("Computing element force vectors...")
        self.element_force_vectors = self._compute_force_vectors_vectorized()
        logger.info(f"✅ Computed {len(self.element_force_vectors)} force vectors.")

        # Convert Ke to sparse format
        #logger.info("Converting stiffness matrices to sparse COO format...")
        #self.element_stiffness_matrices = self._convert_to_sparse(self.element_stiffness_matrices)
        #logger.info("✅ Stiffness matrices successfully converted to sparse format.")

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

        elements = create_elements_batch(self.mesh_dictionary, params_list)

        # Check for missing elements
        if any(el is None for el in elements):
            missing_indices = [i for i, el in enumerate(elements) if el is None]
            logger.warning(f"⚠️ Warning: Missing elements at indices {missing_indices}!")

        return elements

    def _compute_stiffness_matrices_vectorized(self):
        """
        Computes the element stiffness matrices (Ke) using NumPy broadcasting.

        Returns:
            np.ndarray: An array of element stiffness matrices.
        """
        stiffness_matrices = []
        for idx, element in enumerate(self.elements_instances):
            if element is None:
                logger.error(f"❌ Error: Element {idx} is None. Skipping stiffness matrix computation.")
                stiffness_matrices.append(None)
                continue
            try:
                Ke = element.element_stiffness_matrix()
                stiffness_matrices.append(Ke)
            except Exception as e:
                logger.error(f"❌ Error computing stiffness matrix for element {idx}: {e}")
                stiffness_matrices.append(None)
        
        return np.array(stiffness_matrices, dtype=object)

    def _compute_force_vectors_vectorized(self):
        """
        Computes the element force vectors (Fe) using NumPy broadcasting.

        Returns:
            np.ndarray: An array of element force vectors.
        """
        force_vectors = []
        for idx, element in enumerate(self.elements_instances):
            if element is None:
                logger.error(f"❌ Error: Element {idx} is None. Skipping force vector computation.")
                force_vectors.append(None)
                continue
            try:
                Fe = element.element_force_vector()
                force_vectors.append(Fe)
            except Exception as e:
                logger.error(f"❌ Error computing force vector for element {idx}: {e}")
                force_vectors.append(None)
        
        return np.array(force_vectors, dtype=object)

    def _convert_to_sparse(self, matrix_array):
        """
        Converts an array of dense matrices or vectors to sparse COO format.

        Args:
            matrix_array (np.ndarray): Array of dense matrices or vectors.

        Returns:
            list: A list of sparse matrices in COO format or dense vectors.
        """
        if matrix_array is None:
            logger.warning("⚠️ Warning: Attempting to convert NoneType matrix array to sparse format.")
            return []

        return [
            coo_matrix(matrix.reshape(1, -1) if matrix is not None and matrix.ndim == 1 else matrix)
            if matrix is not None else None
            for matrix in np.asarray(matrix_array, dtype=object)
        ]

    def assemble_global_dof_indices(self, element_id):
        """
        Constructs the global DOF indices for an element.

        Args:
            element_id (int): The ID of the element.

        Returns:
            list: A list of global DOF indices associated with the element.
        """
        element_index = np.where(self.mesh_dictionary["element_ids"] == element_id)[0][0]
        node_ids = self.mesh_dictionary["connectivity"][element_index]

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
                logger.warning(f"⚠️ Warning: Skipping validation for missing element {idx}.")
                continue

            if element.Ke is None:
                logger.error(f"❌ Error: Stiffness matrix (Ke) is None for element {idx}.")
                continue

            if element.Fe is None:
                logger.error(f"❌ Error: Force vector (Fe) is None for element {idx}.")
                continue

            assert element.Ke.shape == expected_Ke_shape, (
                f"Element {self.mesh_dictionary['element_ids'][idx]}: Ke shape mismatch. "
                f"Expected {expected_Ke_shape}, got {element.Ke.shape}"
            )
            assert element.Fe.shape == expected_Fe_shape, (
                f"Element {self.mesh_dictionary['element_ids'][idx]}: Fe shape mismatch. "
                f"Expected {expected_Fe_shape}, got {element.Fe.shape}"
            )

        logger.info("✅ Element stiffness matrices and force vectors successfully validated.")
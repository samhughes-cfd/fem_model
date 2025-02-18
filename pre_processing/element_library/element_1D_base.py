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
        self.elements_instances = None
        self.load_array = load_array
        self.dof_per_node = dof_per_node  # Default max DOFs per node = 6

    def _instantiate_elements(self):
        """
        Requests batch element instantiation from `element_factory.py`.

        Returns:
            np.ndarray: Array of instantiated element objects.
        """
        if hasattr(self, 'elements_instances'):
            logger.warning("Skipping element instantiation to prevent recursion.")
            return self.elements_instances  # Return existing elements if already instantiated

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
            logger.warning(f"Warning: Missing elements at indices {missing_indices}!")

        return elements

    def _compute_stiffness_matrices_vectorized(self):
        """
        Computes element stiffness matrices (Ke) in a fully vectorized manner.

        Returns:
            np.ndarray: A 3D NumPy array of shape `(num_elements, dof_per_element, dof_per_element)`,
                        where `num_elements` is the number of elements,
                        and `dof_per_element` is `2 * self.dof_per_node`.
        """
        num_elements = len(self.elements_instances)
        dof_per_element = 2 * self.dof_per_node  # Assuming each element has 2 nodes

        # Initialize a 3D NumPy array to store all stiffness matrices
        stiffness_matrices = np.zeros((num_elements, dof_per_element, dof_per_element))

        # Extract and store stiffness matrices
        for i, element in enumerate(self.elements_instances):
            if element is not None:
                Ke = element.element_stiffness_matrix()
                if isinstance(Ke, np.ndarray) and Ke.shape == (dof_per_element, dof_per_element):
                    stiffness_matrices[i] = Ke  # Store in the array
                else:
                    logger.warning(f"Element {i}: Stiffness matrix shape mismatch {Ke.shape}, expected {(dof_per_element, dof_per_element)}")

        # Convert the entire batch to sparse format (optional)
        stiffness_matrices_sparse = np.array([
            coo_matrix(Ke) for Ke in stiffness_matrices
        ], dtype=object)

        return stiffness_matrices_sparse

    def _compute_force_vectors_vectorized(self):
        """
        Computes element force vectors (Fe) in a fully vectorized manner.

        Returns:
            np.ndarray: A 1D NumPy array (dtype=object) of shape `(num_elements,)` where each element
                        is a dense NumPy array of shape `(dof_per_element,)`.
        """
        num_elements = len(self.elements_instances)
        dof_per_element = 2 * self.dof_per_node  # Assuming each element has 2 nodes

        # ✅ Initialize storage for force vectors (as a list of NumPy arrays)
        force_vectors = np.empty(num_elements, dtype=object)

        # ✅ Extract and store force vectors
        for i, element in enumerate(self.elements_instances):
            if element is not None:
                Fe = element.element_force_vector()
                if isinstance(Fe, np.ndarray) and Fe.shape == (dof_per_element,):
                    force_vectors[i] = Fe  # ✅ Store as a dense 1D array
                else:
                    logger.warning(f"Element {i}: Force vector shape mismatch {Fe.shape}, expected {(dof_per_element,)}")
                    force_vectors[i] = np.zeros(dof_per_element)  # Fallback to zero vector

        return force_vectors  # ✅ Returns a 1D NumPy array of dense vectors

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

        print(f"Element ID: {element_id}, Element Index: {element_index}, Node IDs: {node_ids}")

        global_dof_indices = []
        for node_id in node_ids:
            if node_id < 0:  # Node ID should never be negative
                raise ValueError(f"Invalid node ID detected: {node_id} in element {element_id}")

            # FIXED: Remove -1 since node_id already starts at 0
            start_dof = node_id * self.dof_per_node
            dof_indices = list(range(start_dof, start_dof + self.dof_per_node))

            print(f"Node {node_id}: Start DOF={start_dof}, DOF Indices={dof_indices}")

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
                logger.warning(f"Warning: Skipping validation for missing element {idx}.")
                continue

            if element.Ke is None:
                logger.error(f"Error: Stiffness matrix (Ke) is None for element {idx}.")
                continue

            if element.Fe is None:
                logger.error(f"Error: Force vector (Fe) is None for element {idx}.")
                continue

            assert element.Ke.shape == expected_Ke_shape, (
                f"Element {self.mesh_dictionary['element_ids'][idx]}: Ke shape mismatch. "
                f"Expected {expected_Ke_shape}, got {element.Ke.shape}"
            )
            assert element.Fe.shape == expected_Fe_shape, (
                f"Element {self.mesh_dictionary['element_ids'][idx]}: Fe shape mismatch. "
                f"Expected {expected_Fe_shape}, got {element.Fe.shape}"
            )

        logger.info("Element stiffness matrices and force vectors successfully validated.")
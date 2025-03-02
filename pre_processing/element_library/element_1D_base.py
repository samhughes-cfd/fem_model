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

    def __init__(self, geometry_array, material_array, mesh_dictionary, point_load_array, distributed_load_array, dof_per_node=6):
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
        self.point_load_array = point_load_array
        self.distributed_load_array = distributed_load_array
        self.dof_per_node = dof_per_node
        self.elements_instances = None

    def _instantiate_elements(self):
        """Updated element factory interface with proper parameters"""
        if hasattr(self, 'elements_instances'):
            logger.warning("Skipping element instantiation to prevent recursion.")
            return self.elements_instances

        params_list = np.array([{
            "geometry_array": self.geometry_array,
            "material_array": self.material_array,
            "mesh_dictionary": self.mesh_dictionary,
            "point_load_array": self.point_load_array,
            "distributed_load_array": self.distributed_load_array,
            "element_id": elem_id  # Critical addition
        } for elem_id in self.mesh_dictionary["element_ids"]], dtype=object)

        elements = create_elements_batch(self.mesh_dictionary, params_list)

        # Validation remains unchanged
        if any(el is None for el in elements):
            missing_indices = [i for i, el in enumerate(elements) if el is None]
            logger.warning(f"Missing elements at indices: {missing_indices}")

        return elements

    def _compute_stiffness_matrices_vectorized(self):
        """Handle sparse matrix conversion"""
        stiffness_matrices = super()._compute_stiffness_matrices_vectorized()
        return [mat.tocsr() for mat in stiffness_matrices]  # Convert to CSR for assembly

    def _compute_force_vectors_vectorized(self):
        """Ensure force vector consistency"""
        vectors = super()._compute_force_vectors_vectorized()
        return np.array([v.flatten() for v in vectors], dtype=np.float64)

    def assemble_global_dof_indices(self, element_id):
        """
        Constructs the global DOF indices for an element.

        Args:
            element_id (int): The ID of the element.

        Returns:
            list: A list of global DOF indices associated with the element.
        """
        if element_id not in self.mesh_dictionary["element_ids"]:
            raise ValueError(f"Invalid element_id: {element_id}")

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

        return np.asarray(global_dof_indices, dtype=int) # Returns a NumPy int array
    
    def validate_matrices(self):
        """Updated validation for sparse matrices"""
        expected_Ke_shape = (self.dof_per_node * 2, self.dof_per_node * 2)
        expected_Fe_shape = (self.dof_per_node * 2,)

        for idx, element in enumerate(self.elements_instances):
            if element is None:
                logger.error(f"Null element at index {idx}")
                continue

            # Handle sparse matrices
            Ke = element.Ke.toarray() if isinstance(element.Ke, coo_matrix) else element.Ke
            Fe = element.Fe  # Force vectors remain dense

            if Ke.shape != expected_Ke_shape:
                logger.error(f"Element {idx}: Invalid Ke shape {Ke.shape}")
                
            if Fe.shape != expected_Fe_shape:
                logger.error(f"Element {idx}: Invalid Fe shape {Fe.shape}")

        logger.info("Matrix validation completed")
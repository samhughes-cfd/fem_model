# pre_processing\element_library\element_1D_base.py

import logging
import numpy as np
from scipy.sparse import coo_matrix
from typing import Optional
import os
from pre_processing.element_library.element_factory import create_elements_batch

# Configure Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Debug level logs go to `.log` file

# Console Handler (Minimal terminal output)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Terminal only shows key info/errors
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


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
            point_load_array (np.ndarray): Point loads applied to the system.
            distributed_load_array (np.ndarray): Distributed loads applied to the system.
            dof_per_node (int, optional): Degrees of freedom per node (default: 6).
        """
        self.logger = logger  # Make the logger accessible to child classes
        logger.info("Initializing Element1DBase...")

        self.geometry_array = geometry_array
        self.material_array = material_array
        self.mesh_dictionary = mesh_dictionary
        self.point_load_array = point_load_array
        self.distributed_load_array = distributed_load_array
        self.dof_per_node = dof_per_node
        self.elements_instances = None

    def configure_element_stiffness_logging(self, job_results_dir: Optional[str] = None):
        """Configures logging for element stiffness matrix computations."""
        if job_results_dir:
            stiffness_log_path = os.path.join(job_results_dir, "element_stiffness_matrices.log")
            file_handler = logging.FileHandler(stiffness_log_path, mode="a", encoding="utf-8")  # Append mode
            file_handler.setLevel(logging.DEBUG)  # Full debugging goes here
            file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    def configure_element_force_logging(self, job_results_dir: Optional[str] = None):
        """Configures logging for element force vector computations."""
        if job_results_dir:
            force_log_path = os.path.join(job_results_dir, "element_force_vectors.log")
            file_handler = logging.FileHandler(force_log_path, mode="a", encoding="utf-8")  # Append mode
            file_handler.setLevel(logging.DEBUG)  # Full debugging goes here
            file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

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

        global_dof_indices = []
        for node_id in node_ids:
            if node_id < 0:  # Node ID should never be negative
                raise ValueError(f"Invalid node ID detected: {node_id} in element {element_id}")

            start_dof = node_id * self.dof_per_node
            dof_indices = list(range(start_dof, start_dof + self.dof_per_node))
            global_dof_indices.extend(dof_indices)

        return np.asarray(global_dof_indices, dtype=int)  # Returns a NumPy int array

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
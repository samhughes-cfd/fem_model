# pre_processing/element_library/element_1D_base.py

import logging
import numpy as np
from scipy.sparse import coo_matrix
from typing import Optional, List, Dict
import os
from pre_processing.element_library.element_factory import create_elements_batch
from pre_processing.element_library.base_logger_operator import BaseLoggerOperator

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler for real-time monitoring
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


class Element1DBase:
    """Abstract base class for 1D structural finite elements with hierarchical logging."""

    def __init__(
        self,
        geometry_array: np.ndarray,
        material_array: np.ndarray,
        mesh_dictionary: dict,
        point_load_array: np.ndarray,
        distributed_load_array: np.ndarray,
        dof_per_node: int = 6,
        job_results_dir: Optional[str] = None,  # Added parameter
        element_id: Optional[int] = None  # Added parameter
    ):
        """Initialize base finite element system with enhanced logging support."""
        self.logger = logger
        self.geometry_array = geometry_array
        self.material_array = material_array
        self.mesh_dictionary = mesh_dictionary
        self.point_load_array = point_load_array
        self.distributed_load_array = distributed_load_array
        self.dof_per_node = dof_per_node
        self.elements_instances: List[Element1DBase] = []
        self.job_results_dir = job_results_dir  # Direct storage
        self.element_id = element_id  # Store element ID
        self.logger_operator: Optional[BaseLoggerOperator] = None

        # Initialize logger operator if directory provided
        if self.job_results_dir and self.element_id is not None:
            self.logger_operator = BaseLoggerOperator(self.element_id, self.job_results_dir)
            self._validate_logging_directories()

    def set_logging_directory(self, job_results_dir: str):
        """Configure and validate hierarchical logging output directory."""
        self.job_results_dir = job_results_dir
        
        # Create required subdirectories
        required_dirs = [
            os.path.join(job_results_dir, "element_stiffness_matrices"),
            os.path.join(job_results_dir, "element_force_vectors")
        ]
        for d in required_dirs:
            os.makedirs(d, exist_ok=True)
        
        self._validate_logging_directories()
        logger.info(f"Logging directory configured: {job_results_dir}")

    def _validate_logging_directories(self):
        """Ensure required logging directories exist."""
        required_dirs = [
            os.path.join(self.job_results_dir, "element_stiffness_matrices"),
            os.path.join(self.job_results_dir, "element_force_vectors")
        ]
        for d in required_dirs:
            if not os.path.exists(d):
                raise FileNotFoundError(f"Missing required logging directory: {d}")

    def init_element_logger(self, element_id: int) -> BaseLoggerOperator:
        """Initialize element-specific logging operator with validation."""
        if not self.job_results_dir:
            raise ValueError("Job results directory must be set before initializing element loggers")
        
        logger_op = BaseLoggerOperator(element_id, self.job_results_dir)
        
        # Verify directories were created
        if not os.path.exists(logger_op._get_log_path("stiffness")):
            raise RuntimeError(f"Failed to initialize logging for element {element_id}")
            
        return logger_op

    def _instantiate_elements(self) -> List['Element1DBase']:
        """Batch instantiate elements with proper logging configuration."""
        if self.elements_instances:
            logger.warning("Element instances already exist - returning cached instances")
            return self.elements_instances

        params_list = np.array([{
            "geometry_array": self.geometry_array,
            "material_array": self.material_array,
            "mesh_dictionary": self.mesh_dictionary,
            "point_load_array": self.point_load_array,
            "distributed_load_array": self.distributed_load_array,
            "element_id": elem_id,
            "job_results_dir": self.job_results_dir,  # Critical addition
            "logger_operator": self.init_element_logger(elem_id)
        } for elem_id in self.mesh_dictionary["element_ids"]], dtype=object)

        self.elements_instances = create_elements_batch(self.mesh_dictionary, params_list)
        self._validate_element_creation()
        self._validate_logging_setup()
        return self.elements_instances

    def _validate_element_creation(self):
        """Validate element instantiation and logging setup."""
        missing = [i for i, el in enumerate(self.elements_instances) if el is None]
        if missing:
            logger.error(f"Critical error: Failed to create {len(missing)} elements")
            raise RuntimeError(f"Element creation failed at indices: {missing}")

        # Verify logging operators are initialized
        for idx, element in enumerate(self.elements_instances):
            if not element.logger_operator:
                raise ValueError(f"Element {idx} missing logger operator")

    def _validate_logging_setup(self):
        """Post-instantiation validation of logging infrastructure."""
        if self.job_results_dir:
            # Check for required directories
            required_dirs = [
                os.path.join(self.job_results_dir, "element_stiffness_matrices"),
                os.path.join(self.job_results_dir, "element_force_vectors")
            ]
            for d in required_dirs:
                if not os.path.isdir(d):
                    raise FileNotFoundError(f"Missing logging directory: {d}")

    # Logging wrappers for BaseLoggingOperator interface ------------------------------
    
    def log_matrix(self, category: str, matrix: np.ndarray, metadata: Dict = None):
        """Log numerical matrix with structural metadata.

        Parameters
        ----------
        category : str
            Logging category ('stiffness' or 'force')
        matrix : np.ndarray
            Numerical matrix to log
        metadata : Dict, optional
            Additional metadata fields:
            - 'name': Matrix description
            - 'precision': Display precision
            - 'max_line_width': Formatting width

        Notes
        -----
        Buffers data for batch writing. Actual I/O occurs during flush_logs().
        """
        if self.logger_operator:
            self.logger_operator.log_matrix(category, matrix, metadata)

    def log_text(self, category: str, message: str):
        """Log textual information to category-specific stream.

        Parameters
        ----------
        category : str
            Logging category ('stiffness' or 'force')
        message : str
            Textual message to log
        """
        if self.logger_operator:
            self.logger_operator.log_text(category, message)

    def flush_logs(self, category: Optional[str] = None):
        """Flush logged data to persistent storage.

        Parameters
        ----------
        category : Optional[str], optional
            Specific category to flush, by default flushes all

        Notes
        -----
        Critical for I/O performance in large-scale systems. Batch writing
        reduces filesystem operations by 2-3 orders of magnitude vs per-write.
        """
        if self.logger_operator:
            if category:
                self.logger_operator.flush(category)
            else:
                self.logger_operator.flush_all()

    # Core finite element operations -------------------------------------------
    def _compute_stiffness_matrices_vectorized(self) -> List[coo_matrix]:
        """Compute element stiffness matrices in vectorized form.

        Returns
        -------
        List[coo_matrix]
            List of stiffness matrices in COOrdinate sparse format

        Raises
        ------
        NotImplementedError
            Must be implemented in concrete subclasses

        Notes
        -----
        Expected matrix dimensions: (2*dof_per_node, 2*dof_per_node)
        COO format preferred for assembly efficiency.
        """
        raise NotImplementedError("Stiffness matrix computation not implemented")

    def _compute_force_vectors_vectorized(self) -> np.ndarray:
        """Compute element force vectors in vectorized form.

        Returns
        -------
        np.ndarray
            Force vectors array of shape (n_elements, 2*dof_per_node)

        Raises
        ------
        NotImplementedError
            Must be implemented in concrete subclasses
        """
        raise NotImplementedError("Force vector computation not implemented")

    def assemble_global_dof_indices(self, element_id: int) -> np.ndarray:
        """Compute global DOF indices for element assembly.

        Parameters
        ----------
        element_id : int
            Target element identifier

        Returns
        -------
        np.ndarray
            Global DOF indices array of shape (2*dof_per_node,)

        Raises
        ------
        ValueError
            For invalid element_id or negative node indices

        Notes
        -----
        Assumes consecutive node numbering and fixed dof_per_node.
        Index mapping formula: global_dof = node_id * dof_per_node + local_dof
        """
        if element_id not in self.mesh_dictionary["element_ids"]:
            raise ValueError(f"Invalid element_id: {element_id}")

        element_idx = np.where(self.mesh_dictionary["element_ids"] == element_id)[0][0]
        node_ids = self.mesh_dictionary["connectivity"][element_idx]

        dof_indices = []
        for nid in node_ids:
            if nid < 0:
                raise ValueError(f"Invalid node ID {nid} in element {element_id}")
            start_dof = nid * self.dof_per_node
            dof_indices.extend(range(start_dof, start_dof + self.dof_per_node))

        return np.array(dof_indices, dtype=int)

    def validate_matrices(self):
        """Validate element matrix dimensions and properties.

        Checks:
        1. Stiffness matrix (Ke) is square with proper dimensions
        2. Force vector (Fe) has matching dimension
        3. Stiffness matrix symmetry (warning only)
        4. Positive definiteness of Ke (warning only)

        Notes
        -----
        Logs validation failures to both console and category-specific log files.
        """
        expected_Ke_shape = (self.dof_per_node * 2, self.dof_per_node * 2)
        expected_Fe_shape = (self.dof_per_node * 2,)

        for idx, element in enumerate(self.elements_instances):
            if not element:
                self._log_validation_error(idx, "system", "Null element instance")
                continue

            # Stiffness matrix checks
            if Ke.shape != expected_Ke_shape:
                self._log_validation_error(idx, "stiffness", 
                    f"Shape {Ke.shape} vs {expected_Ke_shape}")

            # Force vector checks
            Fe = element.Fe
            if Fe.shape != expected_Fe_shape:
                self._log_validation_error(idx, "force",
                    f"Shape {Fe.shape} vs {expected_Fe_shape}")

            # Advanced property checks
            if not np.allclose(Ke, Ke.T, atol=1e-6):
                self.log_text("stiffness",
                    f"Element {idx}: Stiffness matrix asymmetry detected")
                
            try:
                np.linalg.cholesky(Ke)
            except np.linalg.LinAlgError:
                self.log_text("stiffness",
                    f"Element {idx}: Stiffness matrix not positive definite")

        logger.info("Matrix validation completed with %d elements checked",
                    len(self.elements_instances))

    def _log_validation_error(self, idx: int, category: str, message: str):
        """Unified validation error logging."""
        full_msg = f"Element {idx} validation error ({category}): {message}"
        logger.error(full_msg)
        self.log_text(category, full_msg)

    # System lifecycle management ----------------------------------------------
    def finalize_system(self):
        """Finalize system processing and ensure data persistence."""
        self._flush_all_element_logs()
        logger.info("System finalized. Results in: %s", self.job_results_dir)
        self._verify_log_integrity()

    def _verify_log_integrity(self):
        """Verify all expected log files were created."""
        if not self.job_results_dir:
            return

        expected_logs = []
        for element in self.elements_instances:
            expected_logs.extend([
                os.path.join(self.job_results_dir, "element_stiffness_matrices", 
                           f"stiffness_element_{element.element_id}.log"),
                os.path.join(self.job_results_dir, "element_force_vectors", 
                           f"force_element_{element.element_id}.log")
            ])

        missing_logs = [log for log in expected_logs if not os.path.exists(log)]
        if missing_logs:
            logger.error(f"Missing {len(missing_logs)} log files")
            raise FileNotFoundError(f"Missing log files: {missing_logs[:3]}...")

    def _flush_all_element_logs(self):
        """Ensure all element logs are persisted to disk."""
        for element in self.elements_instances:
            if element and element.logger_operator:
                element.logger_operator.flush_all()
        logger.debug("Flushed logs for %d elements", len(self.elements_instances))
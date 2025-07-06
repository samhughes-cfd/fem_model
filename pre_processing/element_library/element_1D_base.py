# pre_processing/element_library/element_1D_base.py

import logging
import numpy as np
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import os
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


class Element1DBase(ABC):
    """
    Abstract base class for 1-D structural finite elements.

    Parameters
    ----------
    element_id
        Unique identifier for the element.
    element_dictionary, grid_dictionary, material_dictionary, section_dictionary
        Parsed master dictionaries for the whole model.
    point_load_array, distributed_load_array
        Load tables (may be ``np.empty``).
    job_results_dir
        Directory that already contains ``element_stiffness_matrices`` and
        ``element_force_vectors`` sub-folders.
    dof_per_node
        Number of degrees of freedom per node (default 6).
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        element_id: int,
        element_dictionary: Dict[str, Any],
        grid_dictionary: Dict[str, Any],
        material_dictionary: Dict[str, Any],
        section_dictionary: Dict[str, Any],
        point_load_array: np.ndarray,
        distributed_load_array: np.ndarray,
        job_results_dir: str,
        dof_per_node: int = 6,
    ):
        # --- basic validation ----------------------------------------- #
        if not isinstance(job_results_dir, str):
            raise TypeError("job_results_dir must be a string")
        if not isinstance(element_id, int):
            raise TypeError("element_id must be an integer")

        # --- locate this element’s row -------------------------------- #
        try:
            idx = int(np.where(element_dictionary["ids"] == element_id)[0][0])
        except IndexError as exc:
            raise ValueError(
                f"element_id {element_id} not present in element_dictionary"
            ) from exc

        # ─── build per-element array data access ─────────────────────────── #

        # Connectivity for this element ─ two node indices (int64, shape = (2,))
        node_ids = element_dictionary["connectivity"][idx]                 # [n1, n2]

        # 1) GRID slice  → coordinates only (float64, shape = (2, 3))
        #    rows  : node-0, node-1
        #    cols  : [x, y, z]
        self.grid_array = grid_dictionary["coordinates"][node_ids].astype(
            np.float64, copy=False
        )

        # 2) ELEMENT slice  → id, connectivity, integration orders (int64, shape = (10,))
        self.element_array = np.asarray(
            [
                element_dictionary["ids"][idx],            # element id
                node_ids[0],                               # node-1 id
                node_ids[1],                               # node-2 id
                element_dictionary["integration_orders"]["axial"][idx],
                element_dictionary["integration_orders"]["bending_y"][idx],
                element_dictionary["integration_orders"]["bending_z"][idx],
                element_dictionary["integration_orders"]["shear_y"][idx],
                element_dictionary["integration_orders"]["shear_z"][idx],
                element_dictionary["integration_orders"]["torsion"][idx],
                element_dictionary["integration_orders"]["load"][idx],
            ],
            dtype=np.int64
        )

        #   3) SECTION slice  → cross-section properties (float64, shape = (5,))
        self.section_array = np.asarray(
            [
                section_dictionary["A"][idx],
                section_dictionary["I_x"][idx],
                section_dictionary["I_y"][idx],
                section_dictionary["I_z"][idx],
                section_dictionary["J_t"][idx],
            ],
            dtype=np.float64
        )

        # 4) MATERIAL slice  → material properties (float64, shape = (4,))
        self.material_array = np.asarray(
            [
                material_dictionary["E"][idx],
                material_dictionary["G"][idx],
                material_dictionary["nu"][idx],
                material_dictionary["rho"][idx],
            ],
            dtype=np.float64
        )

        # --- store remaining arguments -------------------------------- #
        self.point_load_array       = point_load_array
        self.distributed_load_array = distributed_load_array
        self.job_results_dir        = job_results_dir
        self.element_id             = element_id
        self.dof_per_node           = dof_per_node

        # optional placeholders for cached results
        self.Ke: Optional[np.ndarray] = None   # stiffness matrix cache
        self.Fe: Optional[np.ndarray] = None   # force vector cache


        # --- logging setup -------------------------------------------- #
        self.logger_operator = BaseLoggerOperator(
            element_id=self.element_id, job_results_dir=self.job_results_dir
        )

        self._assert_logging_ready()
    
    def set_logging_directory(self, job_results_dir: str) -> None:
        """Re-point this element’s log files to a new directory."""
        self.job_results_dir = job_results_dir
        self.logger_operator = BaseLoggerOperator(
            element_id=self.element_id,
            job_results_dir=job_results_dir,
        )
        self._assert_logging_ready()        # ← same check
        logger.info("Logging directory updated: %s", job_results_dir)

    def _assert_logging_ready(self) -> None:
        """Ensure required sub-folders exist and are writable."""
        required = ("element_stiffness_matrices", "element_force_vectors")
        for sub in required:
            path = os.path.join(self.job_results_dir, sub)
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Missing logging directory: {path}")
            if not os.access(path, os.W_OK):
                raise PermissionError(f"Write access denied: {path}")

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

    # Finite element kernels, replaced by subclass

    @abstractmethod
    def element_stiffness_matrix(self) -> np.ndarray:
        """Return local stiffness matrix."""
        raise NotImplementedError

    @abstractmethod
    def element_force_vector(self) -> np.ndarray:
        """Return local equivalent nodal force vector."""
        raise NotImplementedError


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
        eid, n1, n2 = self.element_array[:3]

        if element_id != eid:
            raise ValueError(f"Invalid element_id {element_id} for this object")

        dof_indices = []
        for nid in (n1, n2):
            if nid < 0:
                raise ValueError(f"Invalid node ID {nid} in element {eid}")
            start = nid * self.dof_per_node
            dof_indices.extend(range(start, start + self.dof_per_node))

        return np.asarray(dof_indices, dtype=int)

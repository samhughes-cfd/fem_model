"""
processing/assembly.py

Assembles the global stiffness matrix and force vector for static FEM problems.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from typing import List, Tuple, Optional
import logging
import os

# Configure logging at the module level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a console handler to print logs to the terminal
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

def configure_assembly_logging(job_results_dir: Optional[str] = None):
    """
    Configures logging to write to a file if `job_results_dir` is provided.

    Args:
        job_results_dir (Optional[str]): Directory to save the assembly log file.
    """
    if job_results_dir:
        assembly_log_path = os.path.join(job_results_dir, "assembly.log")
        file_handler = logging.FileHandler(assembly_log_path, mode="w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

def assemble_global_matrices(
    elements: List[object], 
    element_stiffness_matrices: Optional[List[coo_matrix]] = None, 
    element_force_vectors: Optional[List[np.ndarray]] = None, 
    total_dof: int = None,
    job_results_dir: str = None  # Add job_results_dir as an optional argument
) -> Tuple[csr_matrix, np.ndarray]:
    """
    Assembles the global stiffness matrix (K_global) and force vector (F_global) for a static FEM problem.

    Args:
        elements (List[object]): 
            List of element objects implementing `assemble_global_dof_indices()`, returning a NumPy array of DOF indices.
        element_stiffness_matrices (Optional[List[coo_matrix]], default=None): 
            List of sparse COO matrices for element stiffness.
        element_force_vectors (Optional[List[np.ndarray]], default=None): 
            List of 1D NumPy arrays for element force vectors.
        total_dof (int): 
            Total number of degrees of freedom in the system.
        job_results_dir (str): 
            Directory to save the assembly log file.

    Returns:
        Tuple[csr_matrix, np.ndarray]: 
            - `K_global` (csr_matrix): Global stiffness matrix.
            - `F_global` (np.ndarray): Global force vector.
    """
    # Configure logging to write to a file if `job_results_dir` is provided
    configure_assembly_logging(job_results_dir)

    # Log input parameters
    logger.info("🔧 Starting assembly of global matrices...")
    logger.info(f"Number of elements: {len(elements)}")
    logger.info(f"Total DOFs: {total_dof}")
    logger.info(f"Element stiffness matrices provided: {element_stiffness_matrices is not None}")
    logger.info(f"Element force vectors provided: {element_force_vectors is not None}")

    if len(elements) == 0:
        logger.error("❌ Error: elements list is empty, cannot assemble global matrices.")
        raise ValueError("❌ Error: elements list is empty, cannot assemble global matrices.")
    
    if total_dof is None:
        logger.error("❌ Error: total_dof must be specified.")
        raise ValueError("❌ Error: total_dof must be specified.")

    # ✅ Create DOF mappings as a NumPy integer array
    logger.info("🔧 Creating DOF mappings...")
    dof_mappings = np.array(
        [np.array(element.assemble_global_dof_indices(element.element_id), dtype=int) for element in elements]
    )

    # Log DOF mappings
    logger.info("DOF mappings for each element:")
    for i, dof_map in enumerate(dof_mappings):
        logger.info(f"Element {i}: DOF mapping = {dof_map}")

    if dof_mappings.size == 0:
        logger.error("❌ Error: dof_mappings array is empty, no DOF indices available!")
        raise ValueError("❌ Error: dof_mappings array is empty, no DOF indices available!")

    # ✅ Create stiffness matrix in sparse CSR format
    if element_stiffness_matrices is not None:
        logger.info("🔧 Assembling global stiffness matrix (K_global)...")
        Ke_list = np.array(element_stiffness_matrices, dtype=object)
        num_entries = sum(Ke.nnz for Ke in Ke_list)

        K_row = np.zeros(num_entries, dtype=int)
        K_col = np.zeros(num_entries, dtype=int)
        K_data = np.zeros(num_entries, dtype=float)

        offset = 0
        for i, Ke in enumerate(Ke_list):
            nnz = Ke.nnz
            dof_map = dof_mappings[i]
            assert isinstance(dof_map, np.ndarray), f"dof_mappings[{i}] is not a NumPy array!"
            assert np.issubdtype(dof_map.dtype, np.integer), f"dof_mappings[{i}] contains non-integer values!"

            # Log Ke and DOF mapping for debugging
            logger.info(f"Processing element {i}:")
            logger.info(f"  - DOF mapping: {dof_map}")
            logger.info(f"  - Ke shape: {Ke.shape}")
            logger.info(f"  - Ke non-zero entries: {Ke.nnz}")

            K_row[offset:offset + nnz] = dof_map[Ke.row]
            K_col[offset:offset + nnz] = dof_map[Ke.col]
            K_data[offset:offset + nnz] = Ke.data
            offset += nnz

        # Log assembled K_global data
        logger.info("Assembled K_global data:")
        logger.info(f"  - K_row: {K_row}")
        logger.info(f"  - K_col: {K_col}")
        logger.info(f"  - K_data: {K_data}")

        # ✅ Convert to CSR format directly for fast operations
        K_global = coo_matrix((K_data, (K_row, K_col)), shape=(total_dof, total_dof)).tocsr()
        logger.info("✅ Global stiffness matrix (K_global) assembled successfully.")
    else:
        K_global = csr_matrix((total_dof, total_dof))
        logger.info("✅ Empty global stiffness matrix (K_global) created.")

    # ✅ Use a NumPy array for F_global to avoid SparseEfficiencyWarning
    F_global = np.zeros(total_dof, dtype=np.float64)
    logger.info(f"Initialized F_global with shape: {F_global.shape}")

    if element_force_vectors is not None:
        logger.info("🔧 Assembling global force vector (F_global)...")
        for i, Fe in enumerate(element_force_vectors):
            dof_map = dof_mappings[i]

            # Log Fe and DOF mapping for debugging
            logger.info(f"Processing element {i}:")
            logger.info(f"  - DOF mapping: {dof_map}")
            logger.info(f"  - Fe shape before flatten: {Fe.shape}")

            # ✅ Ensure Fe is a 1D NumPy array
            Fe = np.array(Fe, dtype=np.float64).flatten()
            logger.info(f"  - Fe shape after flatten: {Fe.shape}")

            # ✅ Safe update operation: No shape mismatch
            F_global[dof_map] += Fe
            logger.info(f"  - Updated F_global at DOFs: {dof_map}")

        logger.info("✅ Global force vector (F_global) assembled successfully.")

    # Log final K_global and F_global
    logger.info("Final K_global and F_global:")
    logger.info(f"K_global shape: {K_global.shape}")
    logger.info(f"F_global shape: {F_global.shape}")
    logger.info(f"K_global non-zero entries: {K_global.nnz}")
    logger.info(f"F_global: {F_global}")

    return K_global, F_global
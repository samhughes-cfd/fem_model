import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from typing import List, Tuple, Optional
import logging
import os

# ✅ Configure Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Debug level logs go to `.log` file

# ✅ Console Handler (Minimal terminal output)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Terminal only shows key info/errors
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

def configure_assembly_logging(job_results_dir: Optional[str] = None):
    """Configures logging to write detailed logs to a file while suppressing terminal output."""
    logger = logging.getLogger(__name__)
    logger.handlers.clear()  # Remove existing handlers to prevent duplicate logs
    logger.setLevel(logging.DEBUG)  # Full logs go to file

    # ✅ File Handler (Detailed logs)
    if job_results_dir:
        assembly_log_path = os.path.join(job_results_dir, "assembly.log")
        file_handler = logging.FileHandler(assembly_log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        file_handler.setLevel(logging.DEBUG)  # Capture all logs in the file
        logger.addHandler(file_handler)

    # ✅ Console Handler (Suppress detailed output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.CRITICAL)  # Only show CRITICAL messages in the terminal
    logger.addHandler(console_handler)

def assemble_global_matrices(
    elements: List[object], 
    element_stiffness_matrices: Optional[List[coo_matrix]] = None, 
    element_force_vectors: Optional[List[np.ndarray]] = None, 
    total_dof: int = None,
    job_results_dir: str = None  
) -> Tuple[csr_matrix, np.ndarray]:
    """Assembles the global stiffness matrix and force vector while keeping terminal output minimal."""

    configure_assembly_logging(job_results_dir)

    logger.info("🔧 Starting global matrix assembly...")
    logger.debug("This detailed log is written to the assembly.log file")

    if len(elements) == 0:
        logger.error("❌ No elements provided. Assembly aborted.")
        raise ValueError("❌ Error: elements list is empty.")

    if total_dof is None:
        logger.error("❌ total_dof must be specified. Assembly aborted.")
        raise ValueError("❌ Error: total_dof must be specified.")

    logger.info(f"Total DOFs: {total_dof}, Elements count: {len(elements)}")

    # ✅ Create DOF mappings
    try:
        dof_mappings = np.array([
            np.array(element.assemble_global_dof_indices(element.element_id), dtype=int) for element in elements
        ])
    except AttributeError as e:
        logger.error(f"❌ Invalid elements provided. {e}")
        raise ValueError("Invalid elements list. Ensure each element has `assemble_global_dof_indices` method.") from e

    # 🔍 Log DOF mappings (Only in `.log` file)
    if job_results_dir:
        log_file_logger = logging.getLogger(__name__)
        for handler in log_file_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file_logger = handler
                break
        log_file_logger.setLevel(logging.DEBUG)
        
        log_file_logger.stream.write("\nDOF mappings per element:\n")
        for i, dof_map in enumerate(dof_mappings):
            log_file_logger.stream.write(f"Element {i}: DOF mapping = {dof_map}\n")

    # ✅ Assemble global stiffness matrix
    K_global = csr_matrix((total_dof, total_dof))
    if element_stiffness_matrices is not None:
        logger.info("🔧 Assembling global stiffness matrix...")
        Ke_list = np.array(element_stiffness_matrices, dtype=object)
        
        K_entries = []
        for i, Ke in enumerate(Ke_list):
            dof_map = dof_mappings[i]
            K_entries.append((dof_map[Ke.row], dof_map[Ke.col], Ke.data))

        K_row, K_col, K_data = map(np.hstack, zip(*K_entries))
        K_global = coo_matrix((K_data, (K_row, K_col)), shape=(total_dof, total_dof)).tocsr()
        logger.info("✅ Stiffness matrix assembled.")

    # ✅ Assemble force vector
    F_global = np.zeros(total_dof, dtype=np.float64)
    logger.info("🔧 Assembling force vector...")

    if element_force_vectors is not None:
        for i, Fe in enumerate(element_force_vectors):
            dof_map = dof_mappings[i]
            Fe = np.array(Fe, dtype=np.float64).flatten()
            F_global[dof_map] += Fe

        logger.info("✅ Force vector assembled.")

    # ✅ Log structured outputs (Only in `.log` file)
    if job_results_dir:
        if total_dof <= 100:
            df_K = pd.DataFrame(K_global.toarray(), index=range(total_dof), columns=range(total_dof))
            df_F = pd.DataFrame(F_global, index=[f"DOF {i}" for i in range(total_dof)], columns=["Force"])
            
            # 🚀 Write full structured matrices to `.log`, NOT terminal!
            log_file_logger.stream.write("\nFull K_global Matrix:\n" + df_K.to_string(float_format='%.4e') + "\n")
            log_file_logger.stream.write("\nFull F_global Vector:\n" + df_F.to_string(float_format='%.4e') + "\n")
        else:
            # If large, log only non-zero entries
            K_sparse_df = pd.DataFrame({'Row': K_global.row, 'Col': K_global.col, 'Value': K_global.data})
            log_file_logger.stream.write("\nSparse K_global (DOF > 100):\n" + K_sparse_df.to_string(index=False) + "\n")

    logger.info("✅ Assembly complete.")

    return K_global, F_global
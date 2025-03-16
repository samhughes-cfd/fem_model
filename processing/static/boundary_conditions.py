import numpy as np
import logging
import os
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix

# ✅ Configure Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Debug level logs go to `.log` file

# ✅ Console Handler (Minimal terminal output)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Terminal only shows key info/errors
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

def configure_boundary_logging(job_results_dir: str):
    """
    Configures logging to write detailed logs to a file, while keeping the terminal clean.
    
    Args:
        job_results_dir (str): Directory where the boundary conditions log will be stored.
    """
    if job_results_dir:
        boundary_log_path = os.path.join(job_results_dir, "boundary_conditions.log")
        file_handler = logging.FileHandler(boundary_log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Full debugging goes here
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

def apply_boundary_conditions(K_global, F_global, job_results_dir: str = None):
    """
    Applies fixed boundary conditions (first 6 DOFs) to the global stiffness matrix and force vector using the Penalty Method.

    Args:
        K_global (csr_matrix or np.ndarray): 
            Global stiffness matrix of the system. Can be a SciPy CSR matrix or a NumPy array.
        F_global (np.ndarray): 
            Global force vector of the system, expected as a 1D NumPy array.
        job_results_dir (str, optional):
            Directory to store the `.log` file.

    Returns:
        Tuple[csr_matrix, np.ndarray, np.ndarray]:
            - K_mod (csr_matrix): The modified global stiffness matrix in CSR format with boundary conditions applied.
            - F_mod (np.ndarray): The modified global force vector with zero forces at the fixed DOFs.
            - fixed_dofs (np.ndarray): 1D array of global indices where the boundary conditions are applied.
    """
    configure_boundary_logging(job_results_dir)

    logger.info("🔧 Applying fixed boundary conditions...")
    
    # Ensure F_global is a 1D NumPy array
    F_mod = np.asarray(F_global).flatten()
    
    # Define a large penalty value to effectively fix the DOFs.
    large_penalty = 1e36  

    # Define fixed DOFs: fix the first 6 degrees of freedom (indices 0 through 5)
    fixed_dofs = np.arange(6)

    logger.info(f"🔍 Fixed DOFs: {fixed_dofs}")

    # Convert K_global to a mutable LIL format (if it is not already) for efficient row/column modifications.
    if isinstance(K_global, csr_matrix):
        K_mod = K_global.tolil()  # Convert from CSR to LIL for easier modifications
    else:
        K_mod = lil_matrix(K_global)  # Assume K_global is dense and convert it to LIL format

    # 🔍 Log original K_global and F_global (Only in `.log` file)
    if job_results_dir:
        log_file_logger = logging.getLogger(__name__)
        for handler in log_file_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file_logger = handler
                break
        log_file_logger.setLevel(logging.DEBUG)

        if K_global.shape[0] <= 100:
            df_K = pd.DataFrame(K_global.toarray(), index=range(K_global.shape[0]), columns=range(K_global.shape[1]))
            df_F = pd.DataFrame(F_global, index=[f"DOF {i}" for i in range(F_global.shape[0])], columns=["Force"])
            
            log_file_logger.stream.write("\n🔍 Original K_global Matrix:\n" + df_K.to_string(float_format='%.4e') + "\n")
            log_file_logger.stream.write("\n🔍 Original F_global Vector:\n" + df_F.to_string(float_format='%.4e') + "\n")

    # Zero out the rows and columns corresponding to the fixed DOFs.
    K_mod[fixed_dofs, :] = 0
    K_mod[:, fixed_dofs] = 0

    # Set the diagonal entries for each fixed DOF to a large penalty value to enforce the constraint.
    for dof in fixed_dofs:
        K_mod[dof, dof] = large_penalty

    # Set the force vector to zero at the fixed DOFs to prevent any external forces at those locations.
    F_mod[fixed_dofs] = 0

    logger.info("✅ Fixed boundary conditions applied successfully.")

    # Convert the modified stiffness matrix back to CSR format for efficient solving.
    K_mod = K_mod.tocsr()

    # 🔍 Log modified K_global and F_global (Only in `.log` file)
    if job_results_dir:
        if K_mod.shape[0] <= 100:
            df_K_mod = pd.DataFrame(K_mod.toarray(), index=range(K_mod.shape[0]), columns=range(K_mod.shape[1]))
            df_F_mod = pd.DataFrame(F_mod, index=[f"DOF {i}" for i in range(F_mod.shape[0])], columns=["Force"])
            
            log_file_logger.stream.write("\n🔍 Modified K_global Matrix:\n" + df_K_mod.to_string(float_format='%.4e') + "\n")
            log_file_logger.stream.write("\n🔍 Modified F_global Vector:\n" + df_F_mod.to_string(float_format='%.4e') + "\n")
        else:
            K_sparse_df = pd.DataFrame({'Row': K_mod.row, 'Col': K_mod.col, 'Value': K_mod.data})
            log_file_logger.stream.write("\n🔍 Sparse K_global (DOF > 100):\n" + K_sparse_df.to_string(index=False) + "\n")

    return K_mod, F_mod, fixed_dofs
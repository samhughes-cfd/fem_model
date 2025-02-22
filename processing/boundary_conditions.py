import numpy as np
import logging
from scipy.sparse import csr_matrix, lil_matrix

def apply_boundary_conditions(K_global, F_global):
    """
    Applies fixed boundary conditions to the global stiffness matrix and force vector using the Penalty Method.

    Args:
        K_global (csr_matrix or np.ndarray): 
            Global stiffness matrix of the system. Can be a SciPy CSR matrix or a NumPy array.
        F_global (np.ndarray): 
            Global force vector of the system, should be a 1D NumPy array.

    Returns:
        Tuple[csr_matrix, np.ndarray, np.ndarray]:
            - K_mod (csr_matrix): The modified global stiffness matrix in CSR format with BCs applied.
            - F_mod (np.ndarray): The modified global force vector with zero forces at the fixed DOFs.
            - fixed_dofs (np.ndarray): 1D array of global indices where boundary conditions are applied.
    """

    logging.info("Applying cantilever beam boundary conditions.")

    # Ensure F_global is a 1D NumPy array
    F_mod = np.asarray(F_global).flatten()

    # Define a large penalty value for fixed DOFs
    large_penalty = 1e12  

    # Identify fixed DOFs (assuming first 6 are fixed for a cantilever beam)
    fixed_dofs = np.arange(6)

    # Ensure K_global is in a mutable format before modification
    if isinstance(K_global, csr_matrix):
        K_mod = K_global.tolil()  # Convert to LIL format for efficient row/column updates
    else:
        K_mod = lil_matrix(K_global)  # Assume it's dense, convert to LIL format

    # Zero out the rows and columns corresponding to the fixed DOFs
    K_mod[fixed_dofs, :] = 0
    K_mod[:, fixed_dofs] = 0

    # Assign large penalty to the diagonal entries of fixed DOFs
    K_mod.setdiag(large_penalty, k=0)  # Efficiently modifies diagonal

    # Ensure zero external force at the fixed DOFs
    F_mod[fixed_dofs] = 0

    # Convert back to CSR format for efficiency in solving
    K_mod = K_mod.tocsr()

    logging.info("Fixed boundary conditions applied successfully.")

    bc_dofs = fixed_dofs

    return K_mod, F_mod, bc_dofs
# processing\static\boundary_conditions.py

import numpy as np
import logging
from scipy.sparse import csr_matrix, lil_matrix

def apply_boundary_conditions(K_global, F_global):
    """
    Applies fixed boundary conditions (first 6 DOFs) to the global stiffness matrix and force vector using the Penalty Method.

    Args:
        K_global (csr_matrix or np.ndarray): 
            Global stiffness matrix of the system. Can be a SciPy CSR matrix or a NumPy array.
        F_global (np.ndarray): 
            Global force vector of the system, expected as a 1D NumPy array.

    Returns:
        Tuple[csr_matrix, np.ndarray, np.ndarray]:
            - K_mod (csr_matrix): The modified global stiffness matrix in CSR format with boundary conditions applied.
            - F_mod (np.ndarray): The modified global force vector with zero forces at the fixed DOFs.
            - fixed_dofs (np.ndarray): 1D array of global indices where the boundary conditions are applied.
    """
    
    logging.info("Applying fixed boundary conditions to the first 6 DOFs.")

    # Ensure F_global is a 1D NumPy array
    F_mod = np.asarray(F_global).flatten()

    # Define a large penalty value to effectively fix the DOFs.
    large_penalty = 1e12  

    # Define fixed DOFs: fix the first 6 degrees of freedom (indices 0 through 5)
    fixed_dofs = np.arange(6)

    # Convert K_global to a mutable LIL format (if it is not already) for efficient row/column modifications.
    if isinstance(K_global, csr_matrix):
        K_mod = K_global.tolil()  # Convert from CSR to LIL for easier modifications
    else:
        K_mod = lil_matrix(K_global)  # Assume K_global is dense and convert it to LIL format

    # Zero out the rows and columns corresponding to the fixed DOFs.
    # This removes any influence from these DOFs before applying the penalty.
    K_mod[fixed_dofs, :] = 0
    K_mod[:, fixed_dofs] = 0

    # Set the diagonal entries for each fixed DOF to a large penalty value to enforce the constraint.
    for dof in fixed_dofs:
        K_mod[dof, dof] = large_penalty

    # Set the force vector to zero at the fixed DOFs to prevent any external forces at those locations.
    F_mod[fixed_dofs] = 0

    # Convert the modified stiffness matrix back to CSR format for efficient solving.
    K_mod = K_mod.tocsr()

    logging.info("Fixed boundary conditions applied successfully.")

    # Return the modified stiffness matrix, force vector, and the fixed DOF indices.
    return K_mod, F_mod, fixed_dofs
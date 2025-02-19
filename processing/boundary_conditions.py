"""
processing/boundary_conditions.py

Applies a fixed boundary condition to the global stiffness matrix and force vector.
"""

import numpy as np
import logging
from scipy.sparse import csr_matrix

def apply_boundary_conditions(K_global, F_global):
    """
    Applies fixed boundary conditions to the global stiffness matrix and force vector using the Penalty Method.

    This function enforces the boundary conditions for a cantilever beam by constraining the displacements at the fixed 
    degrees of freedom (DOFs), which are assumed to be the first 6 indices (0, 1, 2, 3, 4, 5) in the system. The method 
    works by zeroing out the corresponding rows and columns in the stiffness matrix and then assigning a very large 
    penalty value to the diagonal entries of these fixed DOFs. It also sets the corresponding entries in the force 
    vector to zero to ensure no external forces act on these constrained DOFs.

    Args:
        K_global (csr_matrix or np.ndarray): 
            Global stiffness matrix of the system. This matrix may be provided as a SciPy CSR matrix or a NumPy array.
        F_global (np.ndarray): 
            Global force vector of the system. This should be a 1D NumPy array representing the forces at each DOF.

    Returns:
        Tuple[csr_matrix, np.ndarray]:
            - K_mod (csr_matrix): The modified global stiffness matrix in CSR format with fixed boundary conditions applied.
            - F_mod (np.ndarray): The modified global force vector with zero forces at the fixed DOFs.

    Raises:
        None
    """
    logging.info("Applying cantilever beam boundary conditions.")

    # Ensure F_global is a 1D NumPy array
    F_mod = np.asarray(F_global).flatten()

    # Define penalty value for enforcing zero displacements
    large_penalty = 1e12  

    # Fixed DOFs (first 6 indices)
    fixed_dofs = np.arange(6)

    # Zero out the rows and columns corresponding to the fixed DOFs
    K_global[fixed_dofs, :] = 0
    K_global[:, fixed_dofs] = 0

    # --- Vectorized Diagonal Assignment for Fixed DOFs ---
    # This line ensures that only the submatrix for the fixed DOFs is modified:
    K_global[np.ix_(fixed_dofs, fixed_dofs)] = np.diag(np.full(len(fixed_dofs), large_penalty))

    # Ensure zero external force at the fixed DOFs
    F_mod[fixed_dofs] = 0

    # Convert the modified global stiffness matrix to CSR format for efficient solving
    K_mod = K_global.tocsr()

    logging.info("Fixed boundary conditions applied successfully.")

    return K_mod, F_mod

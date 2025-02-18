"""
processing/boundary_conditions.py

Applies a fixed boundary condition to the global stiffness matrix and force vector.
"""

import numpy as np
import logging
from scipy.sparse import csr_matrix

def apply_boundary_conditions(K_global, F_global):
    """
    Applies a fixed boundary condition for a cantilever beam using the Penalty Method.
    This function modifies only the fixed DOFs, which are the first 6 indices (0, 1, 2, 3, 4, 5).
    """
    logging.info("Applying cantilever beam boundary conditions.")

    # Ensure F_global is a 1D NumPy array
    F_global = np.asarray(F_global).flatten()

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
    F_global[fixed_dofs] = 0

    # Convert the modified global stiffness matrix to CSR format for efficient solving
    K_mod = K_global.tocsr()

    logging.info("Fixed boundary conditions applied successfully.")

    return K_mod, F_global
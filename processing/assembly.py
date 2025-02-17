"""
processing/assembly.py

Assembles the global stiffness matrix and force vector for static FEM problems.
"""

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix

def assemble_global_matrices(elements, element_stiffness_matrices=None, element_force_vectors=None, total_dof=None):
    """
    Assembles the global stiffness matrix (K_global) and force vector (F_global).

    **Optimized Format:**
    - Input stiffness matrices (`element_stiffness_matrices`) must be **`coo_matrix`** for fast assembly.
    - Input force vectors (`element_force_vectors`) must be **1D NumPy arrays** (`(n,)`).
    - Output `K_global` will be **in `lil_matrix` format** for easy boundary condition modification.
    - Output `F_global` will be **a 1D NumPy array** (`(n,)`).
    
    Parameters
    ----------
    elements : list
        List of element objects containing DOF mapping.
    element_stiffness_matrices : list of coo_matrix
        Element stiffness matrices in `coo_matrix` format.
    element_force_vectors : list of ndarray
        Element force vectors as 1D NumPy arrays.
    total_dof : int
        Total number of degrees of freedom in the global system.

    Returns
    -------
    K_global : lil_matrix
        Global stiffness matrix (LIL format for efficient boundary condition application).
    F_global : ndarray
        Global force vector (1D NumPy array).
    """

    # ✅ Convert elements' DOF mappings to a NumPy array for efficient indexing
    dof_mappings = np.array([element.assemble_global_dof_indices(element.element_id) for element in elements])

    # ✅ Ensure inputs are correctly formatted
    if element_stiffness_matrices is not None:
        Ke_list = np.array(element_stiffness_matrices, dtype=object)
    else:
        Ke_list = None

    if element_force_vectors is not None:
        Fe = np.array(element_force_vectors, dtype=object)
    else:
        Fe = None

    # ✅ Vectorized row & column extraction from COO element matrices
    if Ke_list is not None:
        K_row = np.concatenate([Ke.row + dof_mappings[i][0] for i, Ke in enumerate(Ke_list)])
        K_col = np.concatenate([Ke.col + dof_mappings[i][0] for i, Ke in enumerate(Ke_list)])
        K_data = np.concatenate([Ke.data for Ke in Ke_list])

        # ✅ Convert assembled stiffness matrix to `lil_matrix` for efficient boundary condition application
        K_global = coo_matrix((K_data, (K_row, K_col)), shape=(total_dof, total_dof)).tolil()
    else:
        K_global = lil_matrix((total_dof, total_dof))

    # ✅ Assemble force vector as a 1D NumPy array using vectorized addition
    if Fe is not None:
        F_global = np.zeros(total_dof)
        np.add.at(F_global, dof_mappings, Fe)
    else:
        F_global = None

    return K_global, F_global  # ✅ Outputs are optimized for boundary condition application
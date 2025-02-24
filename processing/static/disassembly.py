# processing\disassembly.py

"""
processing/disassembly.py

Disassembles global matrices and vectors back into element-wise quantities in a vectorized manner.
"""

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from typing import List, Tuple

def disassemble_global_matrices(
    elements: List[object], 
    K_mod: csr_matrix, 
    F_mod: np.ndarray, 
    U_global: np.ndarray, 
    R_global: np.ndarray
) -> Tuple[List[coo_matrix], np.ndarray, np.ndarray, np.ndarray]:
    """
    Disassembles the global system into element-wise matrices and vectors using vectorized operations.

    Args:
        elements (List[object]): 
            List of element objects implementing `assemble_global_dof_indices()`, returning a NumPy array of DOF indices.
        K_mod (csr_matrix): 
            Modified global stiffness matrix.
        F_mod (np.ndarray): 
            Modified global force vector.
        U_global (np.ndarray): 
            Global displacement vector.
        R_global (np.ndarray): 
            Global reaction force vector.

    Returns:
        Tuple[List[coo_matrix], np.ndarray, np.ndarray, np.ndarray]:
            - `K_e_mod` (List[coo_matrix]): List of sparse element stiffness matrices.
            - `F_e_mod` (np.ndarray): Stacked element force vectors.
            - `U_e` (np.ndarray): Stacked element displacement vectors.
            - `R_e` (np.ndarray): Stacked element reaction force vectors.

    Raises:
        ValueError: If `elements` is empty.
    """

    if len(elements) == 0:
        raise ValueError("❌ Error: elements list is empty, cannot disassemble matrices.")

    # Construct DOF mappings in a single NumPy array (vectorized)
    dof_mappings = np.array(
        [element.assemble_global_dof_indices(element.element_id) for element in elements], dtype=int
    )

    # Extract element-wise stiffness matrices (Vectorized sparse slicing)
    K_e_mod = [K_mod[dof_map[:, None], dof_map].tocoo() for dof_map in dof_mappings]

    # Vectorized extraction for force, displacement, and reaction vectors
    F_e_mod = np.vstack([F_mod[dof_map] for dof_map in dof_mappings])
    U_e = np.vstack([U_global[dof_map] for dof_map in dof_mappings])
    R_e = np.vstack([R_global[dof_map] for dof_map in dof_mappings])

    return K_e_mod, F_e_mod, U_e, R_e # Element-wise matrices and vectors
"""
processing/assembly.py

Assembles the global stiffness matrix and force vector for static FEM problems.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from typing import List, Tuple, Optional

def assemble_global_matrices(
    elements: List[object], 
    element_stiffness_matrices: Optional[List[coo_matrix]] = None, 
    element_force_vectors: Optional[List[np.ndarray]] = None, 
    total_dof: int = None
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

    Returns:
        Tuple[csr_matrix, np.ndarray]: 
            - `K_global` (csr_matrix): Global stiffness matrix.
            - `F_global` (np.ndarray): Global force vector.

    Raises:
        ValueError: If `elements` is empty or `total_dof` is missing.
        AssertionError: If `dof_mappings` contains non-integer values.
    """

    if len(elements) == 0:
        raise ValueError("❌ Error: elements list is empty, cannot assemble global matrices.")
    
    if total_dof is None:
        raise ValueError("❌ Error: total_dof must be specified.")

    # ✅ Create DOF mappings as a NumPy integer array
    dof_mappings = np.array(
        [np.array(element.assemble_global_dof_indices(element.element_id), dtype=int) for element in elements]
    )

    print(f"Type of dof_mappings: {type(dof_mappings)}, dtype: {dof_mappings.dtype}")
    print(f"dof_mappings shape: {dof_mappings.shape}")

    if dof_mappings.size == 0:
        raise ValueError("❌ Error: dof_mappings array is empty, no DOF indices available!")

    # ✅ Create stiffness matrix in sparse CSR format
    if element_stiffness_matrices is not None:
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

            K_row[offset:offset + nnz] = dof_map[Ke.row]
            K_col[offset:offset + nnz] = dof_map[Ke.col]
            K_data[offset:offset + nnz] = Ke.data
            offset += nnz

        # ✅ Convert to CSR format directly for fast operations
        K_global = coo_matrix((K_data, (K_row, K_col)), shape=(total_dof, total_dof)).tocsr()
    else:
        K_global = csr_matrix((total_dof, total_dof))

    # ✅ Use a NumPy array for F_global to avoid SparseEfficiencyWarning
    F_global = np.zeros(total_dof, dtype=np.float64)

    if element_force_vectors is not None:
        for i, Fe in enumerate(element_force_vectors):
            dof_map = dof_mappings[i]

            # 🔍 Debugging prints
            print(f"Processing element {i}: DOF mapping = {dof_map}")
            print(f"Fe shape before flatten: {Fe.shape}")

            # ✅ Ensure Fe is a 1D NumPy array
            Fe = np.array(Fe, dtype=np.float64).flatten()

            print(f"Fe shape after flatten: {Fe.shape}")

            # ✅ Safe update operation: No shape mismatch
            F_global[dof_map] += Fe

    return K_global, F_global  # ✅ Everything is efficiently stored
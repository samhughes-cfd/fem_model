"""
processing/assembly.py

Assembles the global stiffness matrix and force vector for static FEM problems.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

def assemble_global_matrices(elements, element_mass_matrices=None, element_damping_matrices=None, 
                             element_stiffness_matrices=None, element_force_vectors=None, total_dof=None):
    """
    Generalized function for assembling global mass (M), damping (C), stiffness (K) matrices, 
    and force vector (F) from element-wise contributions.

    Supports:
    - **Static FEM** (`K_global`, `F_global`)
    - **Dynamic FEM** (`M_global`, `C_global`, `K_global`, `F_global`)
    - **Modal Analysis** (`M_global`, `K_global` for eigenvalue problems)

    **Input Requirements:**
    - Element matrices (`element_mass_matrices`, `element_damping_matrices`, `element_stiffness_matrices`)
      **must be provided in `coo_matrix` format** for efficient assembly.
    - Force vectors (`element_force_vectors`) should be **dense NumPy arrays** (`np.ndarray`).

    **Output Format:**
    - Global matrices (`M_global`, `C_global`, `K_global`) are **returned in `csr_matrix` format**
      for efficient numerical operations (solving, matrix-vector multiplication).
    - `F_global` remains a **dense NumPy array** (`np.ndarray`).

    Parameters:
    - elements: List of element objects containing DOF mapping.
    - element_mass_matrices: (Optional) List of element mass matrices (`coo_matrix` form).
    - element_damping_matrices: (Optional) List of element damping matrices (`coo_matrix` form).
    - element_stiffness_matrices: (Optional) List of element stiffness matrices (`coo_matrix` form).
    - element_force_vectors: (Optional) List of element force vectors (`numpy.ndarray` form).
    - total_dof: Total degrees of freedom.

    Returns:
    - M_global: Sparse global mass matrix (`csr_matrix`, None if not provided).
    - C_global: Sparse global damping matrix (`csr_matrix`, None if not provided).
    - K_global: Sparse global stiffness matrix (`csr_matrix`).
    - F_global: Global force vector (`numpy.ndarray`, None if not provided).
    """

    # Convert elements' DOF mappings to a NumPy array for efficient indexing
    dof_mappings = np.array([element.dof_mapping for element in elements])

    # Convert input matrices and vectors to NumPy arrays where applicable
    Fe = np.array(element_force_vectors) if element_force_vectors is not None else None
    Me = np.array(element_mass_matrices) if element_mass_matrices is not None else None
    Ce = np.array(element_damping_matrices) if element_damping_matrices is not None else None

    # Initialize sparse matrix storage (using COO for efficient assembly)
    M_row, M_col, M_data = ([], [], []) if Me is not None else (None, None, None)
    C_row, C_col, C_data = ([], [], []) if Ce is not None else (None, None, None)
    K_row, K_col, K_data = [], [], []

    # Efficiently populate sparse matrix indices & values
    for i, dofs in enumerate(dof_mappings):
        row_indices, col_indices = np.meshgrid(dofs, dofs, indexing='ij')

        # Assemble mass matrix if provided
        if Me is not None:
            M_row.extend(row_indices.ravel())
            M_col.extend(col_indices.ravel())
            M_data.extend(Me[i].data)  # Ensure it’s extracted from `coo_matrix`

        # Assemble damping matrix if provided
        if Ce is not None:
            C_row.extend(row_indices.ravel())
            C_col.extend(col_indices.ravel())
            C_data.extend(Ce[i].data)  # Ensure it’s extracted from `coo_matrix`

        # Assemble stiffness matrix (Ke is already `coo_matrix`)
        if element_stiffness_matrices is not None:
            K_row.extend(element_stiffness_matrices[i].row)
            K_col.extend(element_stiffness_matrices[i].col)
            K_data.extend(element_stiffness_matrices[i].data)

    # Convert lists into sparse matrices (`coo_matrix` → `csr_matrix`)
    M_global = coo_matrix((M_data, (M_row, M_col)), shape=(total_dof, total_dof)).tocsr() if Me is not None else None
    C_global = coo_matrix((C_data, (C_row, C_col)), shape=(total_dof, total_dof)).tocsr() if Ce is not None else None
    K_global = coo_matrix((K_data, (K_row, K_col)), shape=(total_dof, total_dof)).tocsr() if K_data else None

    # Assemble force vector if available
    F_global = np.zeros(total_dof) if Fe is not None else None
    if Fe is not None:
        np.add.at(F_global, dof_mappings, Fe)

    return M_global, C_global, K_global, F_global

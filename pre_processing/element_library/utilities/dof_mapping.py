# pre_processing\element_library\utilities\dof_mapping.py

import numpy as np

def expand_dof_mapping(reduced_array, full_size=12, dof_map=None):
    """
    Expands a reduced array to fit a full DOF system.

    Args:
        reduced_array (ndarray): Reduced array (size depends on dof_map).
            - If `reduced_array` is 2D (e.g., stiffness matrix), its shape should be (num_active_dofs, num_active_dofs).
            - If `reduced_array` is 1D (e.g., force vector), its shape should be (num_active_dofs,).
        full_size (int): Full DOF system size (default 12).
        dof_map (list, optional): Indices where values should be mapped.
            - For stiffness matrix: list of DOF indices to map rows and columns.
            - For force vector: list of DOF indices to map the vector.
            Defaults to mapping all if dof_map is None.

    Returns:
        ndarray: Expanded array (shape: (full_size, full_size) for matrices or (full_size,) for vectors).
    """
    if dof_map is None:
        dof_map = np.arange(full_size)  # Default to all DOFs

    dof_map = np.array(dof_map, dtype=int)  # Convert list to NumPy array

    if np.any(dof_map >= full_size):
        raise ValueError("DOF map contains indices out of bounds.")

    # Vectorized Expansion for Stiffness Matrix (2D)
    if reduced_array.ndim == 2:
        expanded_array = np.zeros((full_size, full_size))
        expanded_array[np.ix_(dof_map, dof_map)] = reduced_array  # Vectorized indexing

    # Vectorized Expansion for Force Vector (1D)
    elif reduced_array.ndim == 1:
        expanded_array = np.zeros(full_size)
        expanded_array[dof_map] = reduced_array  # Vectorized assignment

    else:
        raise ValueError("Reduced array must be either 1D or 2D.")

    return expanded_array

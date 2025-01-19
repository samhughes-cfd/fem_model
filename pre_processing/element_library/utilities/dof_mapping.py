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
        dof_map = list(range(full_size))

    if reduced_array.ndim == 2:
        # Initialize a zero matrix
        expanded_array = np.zeros((full_size, full_size))
        # Map each element of the reduced matrix to the full matrix
        for i, dof_i in enumerate(dof_map):
            for j, dof_j in enumerate(dof_map):
                expanded_array[dof_i, dof_j] = reduced_array[i, j]
    elif reduced_array.ndim == 1:
        # Initialize a zero vector
        expanded_array = np.zeros(full_size)
        # Map each element of the reduced vector to the full vector
        for i, dof_i in enumerate(dof_map):
            expanded_array[dof_i] = reduced_array[i]
    else:
        raise ValueError("Reduced array must be either 1D or 2D.")

    return expanded_array
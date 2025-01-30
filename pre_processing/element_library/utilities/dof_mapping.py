# pre_processing\element_library\utilities\dof_mapping.py

import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

def expand_stiffness_matrix(reduced_Ke, full_size=12, dof_map=None):
    """
    Expands a reduced stiffness matrix (Ke) to fit a full DOF system using efficient vectorized indexing.
    Includes logging to help debug zero entries.

    Args:
        reduced_Ke (ndarray): Reduced stiffness matrix (shape: (num_active_dofs, num_active_dofs)).
        full_size (int): Full DOF system size (default 12).
        dof_map (list or ndarray, optional): Indices where values should be mapped.

    Returns:
        ndarray: Expanded stiffness matrix of shape (full_size, full_size).
    """
    if dof_map is None or len(dof_map) == 0:
        logging.error("DOF mapping must be provided and non-empty.")
        raise ValueError("DOF mapping must be provided and non-empty.")

    dof_map = np.asarray(dof_map, dtype=int)  # Ensure it's a NumPy array

    if np.any(dof_map >= full_size):
        logging.error("DOF map contains indices out of bounds.")
        raise ValueError("DOF map contains indices out of bounds.")

    if reduced_Ke.shape != (len(dof_map), len(dof_map)):
        logging.error("Reduced stiffness matrix size must match DOF map length.")
        raise ValueError("Reduced stiffness matrix size must match DOF map length.")

    # Initialize full-size stiffness matrix to zero
    expanded_Ke = np.zeros((full_size, full_size))

    logging.debug(f"Initializing full stiffness matrix of size {full_size}x{full_size} with zeros.")

    # Directly insert values using advanced NumPy indexing (vectorized)
    expanded_Ke[np.ix_(dof_map, dof_map)] = reduced_Ke  

    logging.info(f"Expanded stiffness matrix updated for DOF indices {dof_map}.")

    # **Vectorized Check for Unexpected Zero Entries**
    mask = np.ix_(dof_map, dof_map)  # Get the relevant submatrix
    zero_mask = expanded_Ke[mask] == 0  # Check for unexpected zeros

    if np.any(zero_mask):
        zero_positions = np.array(np.where(zero_mask)).T
        logging.warning(f"Unexpected zero stiffness found at mapped DOFs: {zero_positions}")

    return expanded_Ke


def expand_force_vector(reduced_Fe, full_size=12, dof_map=None):
    """
    Expands a reduced force vector (Fe) to fit a full DOF system using vectorized mapping.
    Includes logging to help debug zero entries.

    Args:
        reduced_Fe (ndarray): Reduced force vector (size: (num_active_dofs,)).
        full_size (int): Full DOF system size (default 12).
        dof_map (list or ndarray, optional): Indices where values should be mapped.

    Returns:
        ndarray: Expanded force vector of shape (full_size,).
    """
    if dof_map is None or len(dof_map) == 0:
        logging.error("DOF mapping must be provided and non-empty.")
        raise ValueError("DOF mapping must be provided and non-empty.")

    dof_map = np.asarray(dof_map, dtype=int)  # Convert to NumPy array

    if np.any(dof_map >= full_size):
        logging.error("DOF map contains indices out of bounds.")
        raise ValueError("DOF map contains indices out of bounds.")

    if reduced_Fe.shape != (len(dof_map),):
        logging.error("Reduced force vector size must match DOF map length.")
        raise ValueError("Reduced force vector size must match DOF map length.")

    # Initialize full-size force vector to zero
    expanded_Fe = np.zeros(full_size)

    logging.debug(f"Initializing full force vector of size {full_size} with zeros.")

    # Assign reduced values directly using vectorized NumPy indexing
    expanded_Fe[dof_map] = reduced_Fe  

    logging.info(f"Expanded force vector updated for DOF indices {dof_map}.")

    # **Vectorized Check for Unexpected Zero Entries**
    zero_mask = (expanded_Fe[dof_map] == 0)

    if np.any(zero_mask):
        zero_positions = np.array(dof_map)[zero_mask]
        logging.warning(f"Unexpected zero force values found at mapped DOFs: {zero_positions}")

    return expanded_Fe
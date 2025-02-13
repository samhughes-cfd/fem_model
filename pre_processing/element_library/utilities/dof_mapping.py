# pre_processing\element_library\utilities\dof_mapping.py

import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

def log_dof_mapping_details(dof_map_binary, full_size=12):
    """
    Logs the DOF mapping details for each global DOF in a full system.
    Assumes the full 6 DOF per node ordering:
      Node 1: u_x (0), u_y (1), u_z (2), θ_x (3), θ_y (4), θ_z (5)
      Node 2: u_x (6), u_y (7), u_z (8), θ_x (9), θ_y (10), θ_z (11)
      
    The input is a binary list indicating active (1) or inactive (0) for each DOF.
    """
    dof_names = ['u_x', 'u_y', 'u_z', 'θ_x', 'θ_y', 'θ_z']
    
    for global_index in range(full_size):
        node_num = (global_index // 6) + 1
        local_index = global_index % 6
        dof_name = dof_names[local_index]
        status = "ACTIVE" if dof_map_binary[global_index] == 1 else "INACTIVE"
        logging.debug(f"Node {node_num}, global index {global_index}, DOF {dof_name}: {status}")

def expand_stiffness_matrix(reduced_Ke, full_size=12, dof_map_binary=None):
    """
    Expands a reduced stiffness matrix (Ke) to fit a full DOF system using efficient vectorized indexing.
    
    Args:
        reduced_Ke (ndarray): Reduced stiffness matrix of shape (num_active_dofs, num_active_dofs).
        full_size (int): The full DOF system size (default 12).
        dof_map_binary (list or ndarray): A binary vector of length full_size indicating active DOFs.
        
    Returns:
        ndarray: Expanded stiffness matrix of shape (full_size, full_size).
    """
    if dof_map_binary is None or len(dof_map_binary) != full_size:
        logging.error("A binary DOF mapping of length full_size must be provided.")
        raise ValueError("A binary DOF mapping of length full_size must be provided.")
    
    # Log the detailed DOF mapping.
    logging.debug("Detailed DOF mapping (binary):")
    log_dof_mapping_details(dof_map_binary, full_size)
    
    # Convert binary mapping to active indices.
    dof_map = np.where(np.array(dof_map_binary) == 1)[0]
    
    if reduced_Ke.shape != (len(dof_map), len(dof_map)):
        logging.error("Reduced stiffness matrix size must match number of active DOFs.")
        raise ValueError("Reduced stiffness matrix size must match number of active DOFs.")
    
    expanded_Ke = np.zeros((full_size, full_size))
    logging.debug(f"Initializing full stiffness matrix of size {full_size}x{full_size} with zeros.")
    
    # Insert the reduced stiffness matrix at the active DOF positions.
    expanded_Ke[np.ix_(dof_map, dof_map)] = reduced_Ke
    logging.info(f"Expanded stiffness matrix updated for active DOF indices {dof_map}.")
    
    # Check for unexpected zeros in the active submatrix.
    active_mask = np.ix_(dof_map, dof_map)
    zero_mask_active = expanded_Ke[active_mask] == 0
    if np.any(zero_mask_active):
        zero_positions = np.array(np.where(zero_mask_active)).T
        logging.warning(f"⚠️ Zero stiffness found in active DOF submatrix")
        #logging.warning(f"Zero stiffness found in active DOF submatrix at positions: {zero_positions}")
    
    # New check: verify that inactive DOF rows and columns are entirely zero.
    inactive_indices = np.where(np.array(dof_map_binary) == 0)[0]
    for idx in inactive_indices:
        if not np.allclose(expanded_Ke[idx, :], 0):
            logging.error(f"Row {idx} of expanded stiffness matrix is not all zero for inactive DOF.")
        if not np.allclose(expanded_Ke[:, idx], 0):
            logging.error(f"Column {idx} of expanded stiffness matrix is not all zero for inactive DOF.")
    
    return expanded_Ke

def expand_force_vector(reduced_Fe, full_size=12, dof_map_binary=None):
    """
    Expands a reduced force vector (Fe) to fit a full DOF system using vectorized mapping.
    
    Args:
        reduced_Fe (ndarray): Reduced force vector of shape (num_active_dofs,).
        full_size (int): The full DOF system size (default 12).
        dof_map_binary (list or ndarray): A binary vector of length full_size indicating active DOFs.
    
    Returns:
        ndarray: Expanded force vector of shape (full_size,).
    """
    if dof_map_binary is None or len(dof_map_binary) != full_size:
        logging.error("A binary DOF mapping of length full_size must be provided.")
        raise ValueError("A binary DOF mapping of length full_size must be provided.")
    
    logging.debug("Detailed DOF mapping (binary):")
    log_dof_mapping_details(dof_map_binary, full_size)
    
    # Convert binary mapping to active indices.
    dof_map = np.where(np.array(dof_map_binary) == 1)[0]
    
    if reduced_Fe.shape != (len(dof_map),):
        logging.error("Reduced force vector size must match number of active DOFs.")
        raise ValueError("Reduced force vector size must match number of active DOFs.")
    
    expanded_Fe = np.zeros(full_size)
    logging.debug(f"Initializing full force vector of size {full_size} with zeros.")
    
    # Insert the reduced force vector into the active DOF positions.
    expanded_Fe[dof_map] = reduced_Fe
    logging.info(f"Expanded force vector updated for active DOF indices {dof_map}.")
    
    # Check for unexpected zero values in the active DOFs.
    zero_mask_active = (expanded_Fe[dof_map] == 0)
    if np.any(zero_mask_active):
        zero_positions = np.array(dof_map)[zero_mask_active]
        logging.warning(f"⚠️ Unexpected zero force values found in active DOFs")
        #logging.warning(f"⚠️ Unexpected zero force values found in active DOFs at indices: {zero_positions}")
    
    # New check: verify that entries for inactive DOFs are zero.
    inactive_indices = np.where(np.array(dof_map_binary) == 0)[0]
    for idx in inactive_indices:
        if not np.isclose(expanded_Fe[idx], 0):
            logging.error(f"Value at index {idx} of expanded force vector is not zero for inactive DOF.")
    
    return expanded_Fe
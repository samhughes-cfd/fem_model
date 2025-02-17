# pre_processing\element_library\euler_bernoulli\utilities\element_force_vector_3DOF.py

import numpy as np
import logging
from tabulate import tabulate
from pre_processing.element_library.utilities.gauss_quadrature import get_gauss_points
from pre_processing.element_library.utilities.interpolate_loads import interpolate_loads

# Configure logger for this module.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Fixed width for decorative header lines.
MAX_WIDTH = 235

def format_tensor_by_gauss_points(tensor, gp_info, cell_format="{: .4e}"):
    """
    Formats a 3D tensor (shape: (n, rows, cols)) into a multi-column string representation,
    where each column corresponds to a Gauss point.
    """
    n, rows, cols = tensor.shape
    blocks = []
    for i in range(n):
        info = gp_info[i]
        subheader = f"(n={info['n']}), xi={info['xi']}, w={info['w']}"
        block_lines = [subheader]
        for r in range(rows):
            row_vals = " ".join(cell_format.format(x) for x in tensor[i, r, :])
            block_lines.append("[ " + row_vals + " ]")
        blocks.append(block_lines)
    
    block_height = rows + 1  # one subheader plus each row
    col_width = max(len(line) for block in blocks for line in block)
    
    for i in range(n):
        blocks[i] = [line.ljust(col_width) for line in blocks[i]]
    
    sep = " " * 4
    lines = []
    for r in range(block_height):
        lines.append(sep.join(block[r] for block in blocks))
    return "\n".join(lines)

def log_tensor_operation(op_name, tensor, gp_info=None):
    """
    Logs a decorative header along with the given tensor.
    """
    header = "\n".join([
        "*" * MAX_WIDTH,
        f" {op_name} ".center(MAX_WIDTH, "="),
        "*" * MAX_WIDTH
    ])
    if gp_info is not None:
        formatted = format_tensor_by_gauss_points(tensor, gp_info)
        logger.info("\n%s\n%s", header, formatted)
    else:
        logger.info("\n%s\n%s", header, np.array_str(tensor, precision=4, suppress_small=True))

def compute_force_vector(element, n_gauss=3):
    """
    Computes the reduced element force vector (6×1) using Gauss quadrature for a given element.
    
    This function contains two distinct interpolation processes:
    
    1. Load Interpolation:
       - The distributed load (provided via element.load_array) is defined in the physical space.
       - The Gauss points (mapped to physical coordinates via the element geometry) are used to
         interpolate these loads. This is done by calling:
             q_full = interpolate_loads(x_phys_array, element.load_array)
       - If the load vector is provided for all DOFs (e.g., 12 components), it is then filtered using
         the binary DOF mapping to yield a reduced (active) load vector, q_active, of 6 components.
    
    2. Interpolation via Shape Functions:
       - The element's shape functions, evaluated at the Gauss point natural coordinates, yield a tensor
         N_tensor of shape (n,2,6). These functions describe how the field (here, the load) is distributed
         over the element.
       - In the integration step, the shape function matrices are used to “project” the interpolated
         loads at the Gauss points to the nodal degrees of freedom.
    
    Parameters:
      element  : The finite element object with the following attributes and methods:
                 - get_element_index()
                 - mesh_dictionary (with "node_coordinates" and "connectivity")
                 - jacobian_matrix for mapping natural to physical coordinates.
                 - shape_functions(xi) that returns a tuple; the first object is the shape function matrix (N_tensor).
                 - load_array containing the distributed load data.
                 - get_dof_map_binary() returning a binary list for the DOF mapping.
                 - detJ, the determinant of the Jacobian.
      n_gauss : Number of Gauss points to use (default 3).
    
    Returns:
      A NumPy array of shape (6,) representing the computed force vector.
    
    Raises:
      Exception: If any error occurs during computation.
    """
    try:
        # ------------------------------
        # Block 1: Setup and Gauss Point Extraction
        # ------------------------------
        element_index = element.get_element_index()
        header = "\n".join([
            "*" * MAX_WIDTH,
            f" *** Force Vector Computation for Element {element_index} *** ".center(MAX_WIDTH, "="),
            "*" * MAX_WIDTH
        ])
        logger.info(header)
    
        # Retrieve Gauss points and weights; use vectorized extraction of xi.
        gauss_points, weights = get_gauss_points(n=n_gauss, dim=1)
        weights = weights.flatten()
        xi_values = np.array(gauss_points)[:, 0]  # vectorized extraction of natural coordinates.
    
        # Log Gauss points.
        gp_headers = ["n", "xi", "weight"]
        gp_data = [[f"{i+1}", f"{xi:.4f}", f"{w:.4f}"]
                   for i, (xi, w) in enumerate(zip(xi_values, weights))]
        gp_table = tabulate(gp_data, headers=gp_headers, tablefmt="fancy_grid")
        logger.info("Gauss points used for force vector integration:\n%s", gp_table)
    
        # ------------------------------
        # Block 2: Geometry Mapping
        # ------------------------------
        # Compute the element midpoint from nodal coordinates.
        node_coords = element.mesh_dictionary["node_coordinates"]
        connectivity = element.mesh_dictionary["connectivity"][element_index]
        x_mid = node_coords[connectivity].mean(axis=0)
        logger.debug("Midpoint coordinates (x_mid): %s", x_mid)
    
        # Map natural coordinates (xi) to physical coordinates (x_phys) using the Jacobian.
        jacobian_val = element.jacobian_matrix[0, 0]
        x_phys_array = (jacobian_val * xi_values) + x_mid[0]
        logger.debug("Natural coordinates (xi_array): %s", xi_values)
        logger.debug("Physical coordinates (x_array): %s", x_phys_array)
    
        # ------------------------------
        # Block 3: Evaluate Shape Functions (Interpolation via N)
        # ------------------------------
        # Evaluate the shape function matrices for all Gauss points in one call.
        # Here, we extract the first object from the returned tuple.
        N_tensor, _, _ = element.shape_functions(xi=xi_values)
        N_tensor = np.squeeze(N_tensor)  # Ensure shape is (n,2,6)
        if N_tensor.ndim != 3 or N_tensor.shape[1:] != (2, 6):
            raise ValueError(f"Expected shape function tensor of shape (n,2,6), got {N_tensor.shape}")
    
        # Log the shape function matrices for each Gauss point.
        gp_info = [{"n": i+1, "xi": f"{xi:.4f}", "w": f"{w:.4f}"}
                   for i, (xi, w) in enumerate(zip(xi_values, weights))]
        log_tensor_operation("Shape Function Matrix (n,2,6)", N_tensor, gp_info)
    
        # ------------------------------
        # Block 4: Load Interpolation (Separate from N-based Interpolation)
        # ------------------------------
        # This step interpolates the distributed load at the physical coordinates (x_phys_array)
        # using element.load_array. The full load vector (q_full) may contain load values for all DOFs.
        # For a 2-node Euler–Bernoulli element with 3 DOF per node, a full load vector would have 12 components.
        # However, we only want the translational force components (Fx and Fy):
        #   - For Node 1: index 0 (Fx) and index 1 (Fy)
        #   - For Node 2: index 6 (Fx) and index 7 (Fy)
        # We define an active load mapping accordingly, extract these 4 components, and then reduce them
        # to a 2-component load vector by averaging the contributions from the two nodes.
        
        q_full = interpolate_loads(x_phys_array, element.load_array)
        # Ensure q_full is a 2D array: (n_gauss, number_of_columns)
        if q_full.ndim == 1:
            q_full = q_full.reshape(-1, q_full.size)
        
        # Define the active load mapping for a full 12-DOF vector.
        # Only indices corresponding to Fx and Fy are active:
        #   - For Node 1: indices 0 (Fx) and 1 (Fy)
        #   - For Node 2: indices 6 (Fx) and 7 (Fy)
        active_load_mapping = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
        active_indices = np.where(np.array(active_load_mapping) == 1)[0]

        # If q_full is a full 12-component vector, filter to get the 4 active components.
        # Otherwise, assume q_full is already reduced to 4 active DOFs.
        if q_full.shape[1] == len(active_load_mapping):
            q_active_4 = q_full[:, active_indices]  # Shape: (n_gauss, 4)
        else:
            q_active_4 = q_full  # Already reduced to 4 components

        # Reduce the 4-component load vector to a 2-component load vector by averaging the nodal values:
        #   - For Fx: average the value at Node 1 (index 0) with the value at Node 2 (index 2).
        #   - For Fy: average the value at Node 1 (index 1) with the value at Node 2 (index 3).
        n_gp = q_active_4.shape[0]  # Number of Gauss points
        q_active_reduced = np.zeros((n_gp, 2))
        q_active_reduced[:, 0] = 0.5 * (q_active_4[:, 0] + q_active_4[:, 2])  # Combined Fx
        q_active_reduced[:, 1] = 0.5 * (q_active_4[:, 1] + q_active_4[:, 3])  # Combined Fy

        # Log the final active loads (now as a (n_gauss, 1, 2) tensor for formatted logging)
        q_active_display = q_active_reduced[:, np.newaxis, :]  # Reshape for logging purposes
        log_tensor_operation("Interpolated Loads (active DOFs) (n,1,2)", q_active_display, gp_info)
        #logger.info("Active load mapping: %s", active_load_mapping)

        # ------------------------------
        # Block 5: Integration to Compute the Force Vector
        # ------------------------------
        # Use the reduced 2-component load vector in the integration routine.
        Fe_reduced = _integrate_force(weights, N_tensor, q_active_reduced, element.detJ)
        logger.info("Computed force vector: %s", Fe_reduced)
        return Fe_reduced
    
    except Exception as ex:
        logger.exception("Force vector computation failed: %s", ex)
        raise

def _integrate_force(weights, shape_tensor, q_xi_array, detJ):
    """
    Integrates the force vector contributions using multiple methods with robust logging.
    
    For each Gauss point:
      - The shape function matrix (2,6) is transposed to (6,2).
      - Multiplying by the active load vector (2,) yields a (6,) contribution.
      - These contributions are weighted, summed over Gauss points, and scaled by detJ.
    
    This function attempts:
      1. A one-step einsum-based integration.
      2. A three-step einsum-based integration.
      3. A nested loop fallback.
    
    Parameters:
      weights      : 1D NumPy array of Gauss point weights (shape: (n,)).
      shape_tensor : NumPy array of shape (n,2,6) containing evaluated shape functions.
      q_xi_array   : NumPy array of shape (n,6) with active load values.
      detJ         : Scalar representing the determinant of the Jacobian.
    
    Returns:
      A NumPy array of shape (6,) representing the force vector.
    
    Raises:
      RuntimeError: If all integration methods fail.
    """
    try:
        # Method 1: One-step einsum integration.
        try:
            Fe_einsum_1 = _integrate_force_einsum_1(weights, shape_tensor, q_xi_array, detJ)
            logger.info("Force vector computed using one-step einsum integration.")
            return Fe_einsum_1
        except Exception as e:
            logger.error("One-step einsum integration failed: %s", e)
    
        # Method 2: Three-step einsum integration.
        try:
            Fe_einsum_3 = _integrate_force_einsum_3(weights, shape_tensor, q_xi_array, detJ)
            logger.info("Force vector computed using three-step einsum integration.")
            return Fe_einsum_3
        except Exception as e:
            logger.error("Three-step einsum integration failed: %s", e)
    
        # Method 3: Nested loop integration.
        try:
            Fe_nested = _integrate_force_nested(weights, shape_tensor, q_xi_array, detJ)
            logger.info("Force vector computed using nested loop integration.")
            return Fe_nested
        except Exception as e:
            logger.error("Nested loop integration failed: %s", e)
    
        raise RuntimeError("All force vector integration methods failed.")
    
    except Exception as e:
        logger.exception("Error during force vector integration: %s", e)
        raise

def _integrate_force_einsum_1(weights, shape_tensor, q_xi_array, detJ):
    """
    Computes the force vector using a one-step einsum-based integration.
    
    For each Gauss point:
      - Transpose the shape function matrix from (2,6) to (6,2).
      - Multiply by the active load vector (2,) to obtain a (6,) contribution.
      - Sum the contributions weighted by the Gauss weights and scale by detJ.
    
    The einsum call is:
      force_vector = np.einsum("g, gij, gj -> i", weights, N_transposed, q_xi_array) * detJ
    
    Parameters:
      weights      : 1D NumPy array of Gauss point weights.
      shape_tensor : NumPy array of shape (n,2,6).
      q_xi_array   : NumPy array of shape (n,6) with active load values.
      detJ         : Scalar, the determinant of the Jacobian.
    
    Returns:
      A NumPy array of shape (6,) representing the force vector.
    """
    try:
        N_transposed = np.transpose(shape_tensor, axes=(0, 2, 1))
        logger.debug("One-step einsum: N_transposed shape: %s", N_transposed.shape)
        force_vector = np.einsum("g, gij, gj -> i", weights, N_transposed, q_xi_array) * detJ
        logger.debug("One-step einsum: force vector after contraction and scaling:\n%s", force_vector)
        return force_vector
    except Exception as e:
        logger.error("Error during one-step einsum integration: %s", e)
        raise

def _integrate_force_einsum_3(weights, shape_tensor, q_xi_array, detJ):
    """
    Computes the force vector using a three-step (broken down) einsum-based integration.
    
    Steps:
      1. Transpose shape_tensor to obtain N_transposed of shape (n,6,2).
      2. For each Gauss point, compute the product: intermediate = N_transposed dot q_xi_array, yielding (n,6).
      3. Contract over Gauss points using weights and scale by detJ.
    
    Parameters:
      weights      : 1D NumPy array of Gauss point weights.
      shape_tensor : NumPy array of shape (n,2,6).
      q_xi_array   : NumPy array of shape (n,6) with active load values.
      detJ         : Scalar, the determinant of the Jacobian.
    
    Returns:
      A NumPy array of shape (6,) representing the force vector.
    """
    try:
        N_transposed = np.transpose(shape_tensor, axes=(0, 2, 1))
        logger.debug("Three-step einsum: N_transposed shape: %s", N_transposed.shape)
        intermediate = np.einsum("gij, gj -> gi", N_transposed, q_xi_array)
        logger.debug("Three-step einsum: intermediate shape: %s", intermediate.shape)
        weighted_sum = np.einsum("g, gi -> i", weights, intermediate)
        force_vector = weighted_sum * detJ
        logger.debug("Three-step einsum: final force vector:\n%s", force_vector)
        return force_vector
    except Exception as e:
        logger.error("Error during three-step einsum integration: %s", e)
        raise

def _integrate_force_nested(weights, shape_tensor, q_xi_array, detJ):
    """
    Computes the force vector using a nested loop integration approach as a fallback.
    
    For each Gauss point, the (2,6) shape function matrix is transposed to (6,2) and multiplied
    by the active load vector (2,) to yield a (6,) contribution. These contributions are then
    weighted, summed, and scaled by detJ.
    
    Parameters:
      weights      : 1D NumPy array of Gauss point weights.
      shape_tensor : NumPy array of shape (n,2,6).
      q_xi_array   : NumPy array of shape (n,6) with active load values.
      detJ         : Scalar, the determinant of the Jacobian.
    
    Returns:
      A NumPy array of shape (6,) representing the force vector.
    """
    try:
        n_gauss = weights.shape[0]
        force_vector = np.zeros(6)
        for g in range(n_gauss):
            N_transposed = shape_tensor[g].T
            contribution = N_transposed.dot(q_xi_array[g])
            force_vector += weights[g] * contribution
        force_vector *= detJ
        logger.debug("Nested loop: computed force vector:\n%s", force_vector)
        return force_vector
    except Exception as e:
        logger.error("Error during nested loop integration: %s", e)
        raise
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
    Supports 1D, 2D, and 3D tensors.
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
        if tensor.ndim == 1:
            formatted = "[ " + " ".join(f"{x:.4e}" for x in tensor) + " ]"
        else:
            formatted = np.array_str(tensor, precision=4, suppress_small=True)
        logger.info("\n%s\n%s", header, formatted)

def compute_force_vector(element, n_gauss=3):
    """
    Computes the element force vector (shape: (6,)) using Gauss quadrature for a 2-node 
    Euler–Bernoulli beam element in a 3D framework.

    --------------------------------------------------------------------------
    IMPORTANT NOTE ON DOFs & LOAD VECTOR DIMENSIONS:
      - Internally, this code uses an "active_load_mapping" corresponding to a 12-component load vector.
      - Here we extract and average only the x-, y-translational and moment-about-z loads (Fx, Fy, Mz):
            Node 1: indices 0 (Fx), 1 (Fy), 2 (Mz)
            Node 2: indices 6 (Fx), 7 (Fy), 8 (Mz)
      - These 6 components are then averaged down to a 3-component load vector (Fx, Fy, Mz)
        per Gauss point, which is multiplied by a shape function matrix of size (3×6) (from the displacement field)
        to yield a final 6-component force vector for the element.

    --------------------------------------------------------------------------
    WORKFLOW:
      1) Gauss Quadrature: Obtain n_gauss points (xi) and weights.
      2) Geometry & Mapping: Map natural coordinate xi to physical coordinate x.
      3) Evaluate Shape Functions: element.shape_functions(xi=...) should return (N_tensor, dN_dxi, d2N_dxi2)
         where N_tensor has shape (n_gauss, 3, 6).
      4) Distributed Loads Interpolation: Interpolate the user-provided element.load_array at x_phys to get a full load vector,
         then filter out and average the active components to obtain a (n_gauss, 3) load vector (Fx, Fy, Mz).
      5) Integration: Integrate via:
            Force Vector = ∫ N^T * q dx,
         where N^T (for each Gauss point) has shape (6, 3) and q is (3,).

    Parameters
    ----------
    element  : An object representing the element with properties:
               - get_element_index()
               - mesh_dictionary["node_coordinates"], mesh_dictionary["connectivity"]
               - jacobian_matrix (for mapping natural → physical coords)
               - shape_functions(xi=...) -> (N_tensor, dN_dxi, d2N_dxi2)
               - load_array (the distributed load data)
               - detJ (the determinant of the Jacobian, or length scaling)
    n_gauss : int
        Number of Gauss points for 1D integration (default=3).
    
    Returns
    -------
    Fe_reduced : (6,) np.ndarray
        The element force vector corresponding to the active DOFs.
    
    Raises
    ------
    Exception
        If any error occurs during computation.
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

        # Retrieve Gauss points and weights (1D).
        gauss_points, weights = get_gauss_points(n=n_gauss, dim=1)
        weights = weights.flatten()
        xi_values = np.array(gauss_points)[:, 0]  # vectorized extraction

        # Log Gauss points.
        gp_headers = ["n", "xi", "weight"]
        gp_data = [
            [f"{i+1}", f"{xi:.4f}", f"{w:.4f}"]
            for i, (xi, w) in enumerate(zip(xi_values, weights))
        ]
        gp_table = tabulate(gp_data, headers=gp_headers, tablefmt="fancy_grid")
        logger.info("Gauss points used for force vector integration:\n%s", gp_table)

        # ------------------------------
        # Block 2: Geometry Mapping
        # ------------------------------
        node_coords = element.mesh_dictionary["node_coordinates"]
        connectivity = element.mesh_dictionary["connectivity"][element_index]
        x_mid = node_coords[connectivity].mean(axis=0)  # midpoint
        logger.debug("Midpoint coordinates (x_mid): %s", x_mid)

        jacobian_val = element.jacobian_matrix[0, 0]
        x_phys_array = (jacobian_val * xi_values) + x_mid[0]  # 1D mapping along x
        logger.debug("Natural coordinates (xi_array): %s", xi_values)
        logger.debug("Physical coordinates (x_array): %s", x_phys_array)

        # ------------------------------
        # Block 3: Evaluate Shape Functions (N)
        # ------------------------------
        # shape_functions should return (N_tensor, dN_dxi, d2N_dxi2) with N_tensor of shape (n_gauss, 3, 6).
        N_tensor, _, _ = element.shape_functions(xi=xi_values)
        N_tensor = np.squeeze(N_tensor)
        if N_tensor.ndim != 3 or N_tensor.shape[1:] != (3, 6):
            raise ValueError(
                f"Expected shape function tensor of shape (n,3,6), got {N_tensor.shape}"
            )

        # Log shape function matrices for each Gauss point.
        gp_info = [
            {"n": i+1, "xi": f"{xi:.4f}", "w": f"{w:.4f}"}
            for i, (xi, w) in enumerate(zip(xi_values, weights))
        ]
        log_tensor_operation("Shape Function Matrix (n,3,6)", N_tensor, gp_info)

        # ------------------------------
        # Block 4: Load Interpolation
        # ------------------------------

        # Interpolate the user-defined distributed load data at each x_phys.
        q_full = interpolate_loads(x_phys_array, element.load_array)
        if q_full.ndim == 1:
            q_full = q_full.reshape(-1, q_full.size)  # Reshape if 1D

        # Extract the active indices for Fx, Fy, Mz (Node 1 and Node 2)
        active_load_mapping = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]
        active_indices = np.where(np.array(active_load_mapping) == 1)[0]

        # Extract only the active load components
        if q_full.shape[1] == len(active_load_mapping):
            q_active_6 = q_full[:, active_indices]  # shape: (n_gauss, 6)
        else:
            q_active_6 = q_full  # Already reduced

        # Average the contributions from Node 1 and Node 2:
        n_gp = q_active_6.shape[0]
        q_active_reduced = np.zeros((n_gp, 3))  # (Fx, Fy, Mz)
        q_active_reduced[:, 0] = 0.5 * (q_active_6[:, 0] + q_active_6[:, 3])  # Fx avg
        q_active_reduced[:, 1] = 0.5 * (q_active_6[:, 1] + q_active_6[:, 4])  # Fy avg
        q_active_reduced[:, 2] = 0.5 * (q_active_6[:, 2] + q_active_6[:, 5])  # Mz avg

        # Log the new active load vector
        q_active_display = q_active_reduced[:, np.newaxis, :]  # shape: (n_gp, 1, 3)
        log_tensor_operation("Interpolated Loads (active DOFs) (n,1,3)", q_active_display, gp_info)

        # ------------------------------
        # Block 5: Integration
        # ------------------------------
        # With the new shape function matrix (n,3,6) and q_active_reduced (n,3),
        # we compute:  Fe = detJ * sum_over_gp( w_gp * (N^T (gp) * q(gp)) )
        Fe_reduced = _integrate_force(weights, N_tensor, q_active_reduced, element.detJ)
        return Fe_reduced

    except Exception as ex:
        logger.exception("Force vector computation failed: %s", ex)
        raise

def _integrate_force(weights, shape_tensor, q_xi_array, detJ):
    """
    Computes the force vector using multiple integration methods and logs all results.
    """
    force_vectors = {}  # Dictionary to store results from each method

    try:
        # Method 1: One-step einsum
        try:
            force_vectors["One-step Einsum"] = _integrate_force_einsum_1(weights, shape_tensor, q_xi_array, detJ)
        except Exception as e:
            logger.error("One-step einsum integration failed: %s", e)

        # Method 2: Three-step einsum
        try:
            force_vectors["Three-step Einsum"] = _integrate_force_einsum_3(weights, shape_tensor, q_xi_array, detJ)
        except Exception as e:
            logger.error("Three-step einsum integration failed: %s", e)

        # Method 3: Nested loop
        try:
            force_vectors["Nested Loop"] = _integrate_force_nested(weights, shape_tensor, q_xi_array, detJ)
        except Exception as e:
            logger.error("Nested loop integration failed: %s", e)

        # Return the first successful force vector (prioritizing einsum-1)
        for method_name in ["One-step Einsum", "Three-step Einsum", "Nested Loop"]:
            if method_name in force_vectors:
                return force_vectors[method_name]

        raise RuntimeError("All force vector integration methods failed.")

    except Exception as e:
        logger.exception("Error during force vector integration: %s", e)
        raise

def _integrate_force_einsum_1(weights, shape_tensor, q_xi_array, detJ):
    """
    One-step einsum integration:
      force_vector = detJ * sum_over_g( w_g * N^T(g) * q(g) ).
    Here, shape_tensor is of shape (n_gauss, 3, 6) and q_xi_array is (n_gauss, 3).
    We first compute N_transposed = transpose(shape_tensor, axes=(0,2,1)) with shape (n_gauss, 6, 3).
    """
    try:
        N_transposed = np.transpose(shape_tensor, axes=(0, 2, 1))  # (n_gauss, 6, 3)
        force_vector = np.einsum("g, gij, gj -> i", weights, N_transposed, q_xi_array) * detJ
        log_tensor_operation("Force Vector (One-step Einsum)", force_vector, gp_info=None)
        return force_vector
    except Exception as e:
        logger.error("Error during one-step einsum integration: %s", e)
        raise

def _integrate_force_einsum_3(weights, shape_tensor, q_xi_array, detJ):
    """
    Three-step einsum integration to illustrate the decomposition:
      1) intermediate = sum_over_j( N^T(g)[i,j] * q(g)[j] )
      2) weighted     = w_g * intermediate
      3) sum over g, multiply by detJ
    """
    try:
        N_transposed = np.transpose(shape_tensor, axes=(0, 2, 1))  # (n_gauss, 6, 3)
        intermediate = np.einsum("gij, gj -> gi", N_transposed, q_xi_array)   # (n_gauss, 6)
        weighted_sum = np.einsum("g, gi -> i", weights, intermediate)         # (6,)
        force_vector = weighted_sum * detJ
        log_tensor_operation("Force Vector (Three-step Einsum)", force_vector, gp_info=None)
        return force_vector
    except Exception as e:
        logger.error("Error during three-step einsum integration: %s", e)
        raise

def _integrate_force_nested(weights, shape_tensor, q_xi_array, detJ):
    """
    Nested loop integration:
      force_vector = detJ * sum_g( w_g * (N^T(g) * q_xi_array[g]) ).
    """
    try:
        n_gauss = weights.shape[0]
        force_vector = np.zeros(6)
        for g in range(n_gauss):
            N_transposed = shape_tensor[g].T  # shape (6, 3)
            contribution = N_transposed.dot(q_xi_array[g])  # (6,)
            force_vector += weights[g] * contribution
        force_vector *= detJ
        log_tensor_operation("Force Vector (Nested Loop)", force_vector, gp_info=None)
        return force_vector
    except Exception as e:
        logger.error("Error during nested loop integration: %s", e)
        raise

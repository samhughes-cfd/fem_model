# pre_processing\element_library\euler_bernoulli\utilities\element_stiffness_matrix_3DOF.py

import numpy as np
import logging
from tabulate import tabulate
from pre_processing.element_library.utilities.gauss_quadrature import get_gauss_points

# Configure logger for this module.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Use a fixed width for all decorative header lines.
MAX_WIDTH = 235

def format_tensor_by_gauss_points(tensor, gp_info, cell_format="{: .4e}"):
    """
    Formats a 3D tensor (with shape (n, rows, cols)) into a multi-column string
    representation, where each column corresponds to a Gauss point.

    For each Gauss point, a subheader is included showing the Gauss point index,
    its coordinate (xi), and weight (w). Each row of the tensor for that Gauss point
    is printed using the specified cell format.

    The columns are left-justified and separated by four spaces.

    Parameters:
      tensor      : NumPy ndarray of shape (n, rows, cols) containing the tensor data.
      gp_info     : List of dictionaries containing Gauss point information for each n.
                    Each dictionary must include the keys "n", "xi", and "w".
      cell_format : (Optional) A string format specifier for each cell value. Default is "{: .4e}".

    Returns:
      A formatted string with side-by-side columns for each Gauss point.
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

    If Gauss point information is provided via gp_info, the tensor is formatted
    using `format_tensor_by_gauss_points()`. Otherwise, the tensor is logged
    directly using NumPy's array_str().

    Parameters:
      op_name : String describing the operation, used in the header.
      tensor  : NumPy array to be logged.
      gp_info : (Optional) List of dictionaries with Gauss point information, passed to the formatter.
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

def compute_stiffness_matrix(element, n_gauss=3):
    """
    Computes the reduced element stiffness matrix (6x6) using Gauss quadrature for a given element.

    This function performs the following steps:
      1. Logs a header indicating the element being processed.
      2. Retrieves Gauss points and weights, then logs them in a formatted table.
      3. Computes the shape functions and their derivatives with respect to ξ (dN_dxi)
         at the Gauss points and logs the resulting tensor.
      4. Retrieves the material stiffness matrix D and logs it with its shape.
      5. Transposes the dN_dxi tensor and logs the transposed version.
      6. Computes the stiffness matrix using three different methods:
         - A one-step einsum-based approach.
         - A three-step einsum-based approach.
         - A nested loop fallback method.
      7. Logs the computed stiffness matrices from each method.

    Parameters:
      element  : An object representing the finite element. It is expected to have:
                 - attribute `element_id` for identification.
                 - method `shape_functions(xi)` that returns a tuple where the second item is dN_dxi.
                 - method `material_stiffness_matrix()` that returns the material stiffness matrix D.
                 - attribute `detJ` representing the determinant of the Jacobian.
      n_gauss : (Optional) Integer specifying the number of Gauss points to use. Default is 3.

    Returns:
      The stiffness matrix computed using the one-step einsum-based method (6x6 NumPy array).

    Raises:
      ValueError: If the determinant of the Jacobian (detJ) is zero.
      Other exceptions are logged and re-raised.
    """
    try:
        element_id = getattr(element, "element_id", "N/A")
        element_header = "\n".join([
            "*" * MAX_WIDTH,
            f" *** Processing Element {element_id} *** ".center(MAX_WIDTH, "="),
            "*" * MAX_WIDTH
        ])
        logger.info(element_header)

        # Retrieve Gauss points and weights.
        gauss_points, weights = get_gauss_points(n=n_gauss, dim=1)
        weights = weights.flatten()
        xi_values = [xi[0] for xi in gauss_points]

        # Log Gauss points as a table.
        gp_headers = ["n", "xi", "weight"]
        gp_data = [[f"{i+1}", f"{xi:.4f}", f"{w:.4f}"] 
                   for i, (xi, w) in enumerate(zip(xi_values, weights))]
        gp_table = tabulate(gp_data, headers=gp_headers, tablefmt="fancy_grid")
        logger.info("\n%s", gp_table)

        # Compute shape functions.
        # (We assume element.shape_functions returns a tuple where the second item is dN_dxi.)
        _, dN_dxi_tensor, _ = element.shape_functions(xi=xi_values)
        # Instead of logging the xi values and tensor shape separately, we log dN_dxi using our helper.
        gp_info = [{"n": i+1, "xi": f"{xi:.4f}", "w": f"{w:.4f}"} 
                   for i, (xi, w) in enumerate(zip(xi_values, weights))]
        log_tensor_operation("dN_dxi (n, 2, 6)", dN_dxi_tensor, gp_info)

        # Log the material stiffness matrix D with a header including its shape.
        D = element.material_stiffness_matrix()
        log_tensor_operation("Material Stiffness Matrix D", D)
        if np.linalg.cond(D) > 1e10:
            logger.warning("Material stiffness matrix is nearly singular!")

        # Compute and log the transpose of dN_dxi (dN_dxi_T with shape (n, 6, 2)).
        dN_dxi_T_tensor = np.transpose(dN_dxi_tensor, axes=(0, 2, 1))
        log_tensor_operation("dN_dxi_T (n, 6, 2)", dN_dxi_T_tensor, gp_info)

        if np.isclose(element.detJ, 0):
            raise ValueError("detJ is zero, which will cause singular integration.")

        # Compute the reduced stiffness matrix using einsum 1 step.
        Ke_einsum_1 = _integrate_stiffness_einsum_1(weights, dN_dxi_tensor, dN_dxi_T_tensor, D, element.detJ)
        # Compute the reduced stiffness matrix using einsum 3 step.
        Ke_einsum_3 = _integrate_stiffness_einsum_3(weights, dN_dxi_tensor, dN_dxi_T_tensor, D, element.detJ)
        # Also compute using a nested loop fallback.
        Ke_nested = _integrate_stiffness_nested(weights, dN_dxi_tensor, dN_dxi_T_tensor, D, element.detJ)

        # Log the computed stiffness matrices.
        log_tensor_operation(f"Ke_(element {element_id}) - einsum_1", Ke_einsum_1)
        log_tensor_operation(f"Ke_(element {element_id}) - einsum_3", Ke_einsum_3)
        log_tensor_operation(f"Ke_(element {element_id}) - nested loop", Ke_nested)
        return Ke_einsum_1

    except Exception as e:
        logger.error("Stiffness matrix computation failed: %s", e)
        raise

def _integrate_stiffness_einsum_1(weights, dN_dxi_tensor, dN_dxi_T_tensor, D, detJ):
    """
    Computes the stiffness matrix using a one-step einsum-based integration.

    This function performs the integration in a single einsum call:
        np.einsum("gmk,kn,gnj,g->mj", dN_dxi_T_tensor, D, dN_dxi_tensor, weights) * detJ
    where the indices represent:
        - g: Gauss point index
        - m, j: stiffness matrix indices (6 each)
        - k, n: intermediate indices corresponding to the dimensions of D and dN_dxi.

    Parameters:
      weights          : 1D NumPy array of Gauss point weights (shape: (g,)).
      dN_dxi_tensor    : 3D NumPy array with shape (g, 2, 6) representing derivatives of shape functions with respect to ξ.
      dN_dxi_T_tensor  : 3D NumPy array with shape (g, 6, 2), the transpose of dN_dxi_tensor.
      D                : 2D NumPy array of shape (2, 2), the material stiffness matrix.
      detJ             : Scalar value representing the determinant of the Jacobian.

    Returns:
      A 6x6 NumPy array representing the computed stiffness matrix.
    """
    stiffness_matrix = np.einsum("gmk,kn,gnj,g->mj", dN_dxi_T_tensor, D, dN_dxi_tensor, weights) * detJ
    return stiffness_matrix

def _integrate_stiffness_einsum_3(weights, dN_dxi_tensor, dN_dxi_T_tensor, D, detJ):
    """
    Computes the stiffness matrix using a three-step einsum-based integration.

    This method decomposes the integration into three explicit steps:
      1. Compute an intermediate tensor:
         Intermediate1 = np.einsum("gmk,kn->gmn", dN_dxi_T_tensor, D)
      2. Multiply the intermediate tensor by dN_dxi_tensor:
         Intermediate2 = np.einsum("gmn,gnj->gmj", Intermediate1, dN_dxi_tensor)
      3. Integrate over the Gauss points:
         stiffness_matrix = np.einsum("g,gmj->mj", weights, Intermediate2) * detJ

    Each intermediate result is logged for debugging purposes.

    Parameters:
      weights          : 1D NumPy array of Gauss point weights (shape: (g,)).
      dN_dxi_tensor    : 3D NumPy array of shape (g, 2, 6) with derivatives of shape functions.
      dN_dxi_T_tensor  : 3D NumPy array of shape (g, 6, 2), the transpose of dN_dxi_tensor.
      D                : 2D NumPy array of shape (2, 2) representing the material stiffness matrix.
      detJ             : Scalar value, the determinant of the Jacobian.

    Returns:
      A 6x6 NumPy array representing the computed stiffness matrix.
    """
    n_gauss = weights.shape[0]
    
    # Step 1: Compute Intermediate1 = dN_dxi_T_tensor * D.
    intermediate1 = np.einsum("gmk,kn->gmn", dN_dxi_T_tensor, D)
    log_tensor_operation("Intermediate1 = dN_dxi_T * D (n, 6, 2)",
                         intermediate1,
                         [{"n": i+1, "xi": "", "w": ""} for i in range(n_gauss)])
    
    # Step 2: Compute Intermediate2 = Intermediate1 * dN_dxi_tensor.
    intermediate2 = np.einsum("gmn,gnj->gmj", intermediate1, dN_dxi_tensor)
    log_tensor_operation("Intermediate2 = Intermediate1 * dN_dxi (n, 6, 6)",
                         intermediate2,
                         [{"n": i+1, "xi": "", "w": ""} for i in range(n_gauss)])
    
    # Step 3: Final integration over Gauss points.
    stiffness_matrix = np.einsum("g,gmj->mj", weights, intermediate2) * detJ
    return stiffness_matrix

def _integrate_stiffness_nested(weights, dN_dxi_tensor, dN_dxi_T_tensor, D, detJ):
    """
    Computes the stiffness matrix using a nested loop integration approach as a fallback.

    This method explicitly loops over Gauss points and the indices of the stiffness matrix,
    accumulating the contributions from each Gauss point.

    Parameters:
      weights          : 1D NumPy array of Gauss point weights (shape: (g,)).
      dN_dxi_tensor    : 3D NumPy array of shape (g, 2, 6) representing derivatives of shape functions.
      dN_dxi_T_tensor  : 3D NumPy array of shape (g, 6, 2), the transpose of dN_dxi_tensor.
      D                : 2D NumPy array of shape (2, 2) representing the material stiffness matrix.
      detJ             : Scalar value, the determinant of the Jacobian.

    Returns:
      A 6x6 NumPy array representing the computed stiffness matrix.
    """
    n_gauss = weights.shape[0]
    stiffness_matrix = np.zeros((6, 6))
    for g in range(n_gauss):
        for i in range(6):
            for j in range(6):
                sum_term = 0.0
                for a in range(2):
                    for b in range(2):
                        sum_term += dN_dxi_T_tensor[g, i, a] * D[a, b] * dN_dxi_tensor[g, b, j]
                stiffness_matrix[i, j] += weights[g] * sum_term * detJ
    return stiffness_matrix

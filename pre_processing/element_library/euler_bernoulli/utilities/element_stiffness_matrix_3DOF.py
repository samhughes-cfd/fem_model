# pre_processing\element_library\euler_bernoulli\utilities\element_stiffness_matrix_3DOF.py

# -*- coding: utf-8 -*-
"""
element_stiffness_matrix_3DOF.py

This module provides functionality for computing the 6×6 element stiffness matrix of
an Euler–Bernoulli beam element with 3 DOFs per node (axial displacement, transverse displacement,
and rotation). The shape-function derivative tensor is assumed to have shape (n_gauss, 3, 6),
and the material stiffness matrix D is taken to be 3×3, typically containing:
    - EA  : axial rigidity
    - (0) : shear rigidity term (set to zero if classical EB is used)
    - EI  : bending rigidity

Example 3×3 D matrix for Euler–Bernoulli (no shear):
    D = [ [EA,  0,   0  ],
          [ 0,  0,   0  ],
          [ 0,  0,   EI ] ]

Author: Your Name
Date: YYYY-MM-DD
"""

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

    Parameters
    ----------
    tensor : ndarray of shape (n, rows, cols)
        The tensor data to be formatted.
    gp_info : list of dict
        List of dictionaries containing Gauss point info, each with keys "n", "xi", and "w".
    cell_format : str, optional
        A Python format string for each cell value. Default is "{: .4e}".

    Returns
    -------
    str
        A formatted string with side-by-side columns for each Gauss point.
    """
    # -- Determine shapes --
    n, rows, cols = tensor.shape

    # -- Build blocks of text, each block corresponding to one Gauss point --
    blocks = []
    for i in range(n):
        info = gp_info[i]
        subheader = f"(n={info['n']}), xi={info['xi']}, w={info['w']}"
        block_lines = [subheader]  # first line in this column
        for r in range(rows):
            row_vals = " ".join(cell_format.format(x) for x in tensor[i, r, :])
            block_lines.append("[ " + row_vals + " ]")
        blocks.append(block_lines)

    # -- Find the block dimensions --
    block_height = rows + 1  # one subheader plus each row
    col_width = max(len(line) for block in blocks for line in block)

    # -- Pad each line in each block to have equal width, for alignment --
    for i in range(n):
        blocks[i] = [line.ljust(col_width) for line in blocks[i]]

    # -- Join blocks side-by-side with a spacing separator --
    sep = " " * 4
    lines = []
    for r in range(block_height):
        line_segments = [block[r] for block in blocks]
        lines.append(sep.join(line_segments))

    return "\n".join(lines)


def log_tensor_operation(op_name, tensor, gp_info=None):
    """
    Logs a decorative header and the given tensor to the logger.

    If Gauss point information is provided via gp_info, the tensor is formatted
    using `format_tensor_by_gauss_points()`. Otherwise, the tensor is logged
    directly using NumPy's array_str().

    Parameters
    ----------
    op_name : str
        String describing the operation, used in the log header.
    tensor : ndarray
        NumPy array to be logged.
    gp_info : list of dict, optional
        Gauss point information, passed to the formatter if available.
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
    Computes the reduced element stiffness matrix (6×6) using Gauss quadrature for a given element.

    Steps:
    ------
      1. Logs a header indicating the element being processed.
      2. Retrieves Gauss points and weights, then logs them in a formatted table.
      3. Computes the shape functions and their derivatives w.r.t. ξ (dN_dxi)
         at the Gauss points and logs the resulting (n,3,6) tensor.
      4. Retrieves the material stiffness matrix D (3×3) and logs it with its shape.
      5. Transposes the dN_dxi tensor to (n,6,3) and logs the transposed version.
      6. Computes the stiffness matrix using three different methods:
         - A one-step einsum-based approach.
         - A three-step einsum-based approach.
         - A nested-loop fallback method.
      7. Logs the computed stiffness matrices from each method.

    Parameters
    ----------
    element : object
        An object representing the finite element. It must have:
         - attribute `element_id` for identification.
         - method `shape_functions(xi)` returning a tuple, where the second item is dN_dxi.
           (We expect dN_dxi to have shape (n, 3, 6).)
         - method `material_stiffness_matrix()` returning a 3×3 material stiffness matrix D.
         - attribute `detJ` representing the determinant of the Jacobian.
    n_gauss : int, optional
        Number of Gauss points to use (default=3).

    Returns
    -------
    ndarray of shape (6,6)
        The stiffness matrix computed using the one-step einsum-based method.

    Raises
    ------
    ValueError
        If the determinant of the Jacobian (detJ) is zero.
    Exception
        Any other exceptions are logged and re-raised.
    """
    try:
        # ------------------------------------------------------------------
        # (1) LOG ELEMENT HEADER
        # ------------------------------------------------------------------
        element_id = getattr(element, "element_id", "N/A")
        element_header = "\n".join([
            "*" * MAX_WIDTH,
            f" *** Processing Element {element_id} *** ".center(MAX_WIDTH, "="),
            "*" * MAX_WIDTH
        ])
        logger.info(element_header)

        # ------------------------------------------------------------------
        # (2) GAUSS POINTS & WEIGHTS
        # ------------------------------------------------------------------
        gauss_points, weights = get_gauss_points(n=n_gauss, dim=1)
        weights = weights.flatten()  # shape (n_gauss,)
        xi_values = [xi[0] for xi in gauss_points]

        # Log Gauss points as a table
        gp_headers = ["n", "xi", "weight"]
        gp_data = [
            [f"{i+1}", f"{xi:.4f}", f"{w:.4f}"] 
            for i, (xi, w) in enumerate(zip(xi_values, weights))
        ]
        gp_table = tabulate(gp_data, headers=gp_headers, tablefmt="fancy_grid")
        logger.info("\n%s", gp_table)

        # ------------------------------------------------------------------
        # (3) SHAPE FUNCTIONS & DERIVATIVES
        # ------------------------------------------------------------------
        # We assume element.shape_functions returns (N_matrix, dN_dxi_tensor, d2N_dxi2_matrix)
        _, dN_dxi_tensor, _ = element.shape_functions(xi=xi_values)

        # Log the dN_dxi tensor as (n, 3, 6)
        gp_info = [
            {"n": i+1, "xi": f"{xi:.4f}", "w": f"{w:.4f}"}
            for i, (xi, w) in enumerate(zip(xi_values, weights))
        ]
        log_tensor_operation("dN_dxi (n, 3, 6)", dN_dxi_tensor, gp_info)

        # ------------------------------------------------------------------
        # (4) MATERIAL STIFFNESS MATRIX D (3×3)
        # ------------------------------------------------------------------
        D = element.material_stiffness_matrix()
        log_tensor_operation("Material Stiffness Matrix D", D)

        # Optionally check if near-singular
        if np.linalg.cond(D) > 1e10:
            logger.warning("Material stiffness matrix is nearly singular!")

        # ------------------------------------------------------------------
        # (5) TRANSPOSE dN_dxi => (n, 6, 3)
        # ------------------------------------------------------------------
        dN_dxi_T_tensor = np.transpose(dN_dxi_tensor, axes=(0, 2, 1))
        log_tensor_operation("dN_dxi_T (n, 6, 3)", dN_dxi_T_tensor, gp_info)

        # Check determinant of Jacobian
        if np.isclose(element.detJ, 0):
            raise ValueError("detJ is zero, which will cause singular integration.")

        # ------------------------------------------------------------------
        # (6) COMPUTE STIFFNESS VIA MULTIPLE METHODS
        # ------------------------------------------------------------------
        # (a) Einsum 1-step
        Ke_einsum_1 = _integrate_stiffness_einsum_1(
            weights, dN_dxi_tensor, dN_dxi_T_tensor, D, element.detJ
        )
        # (b) Einsum 3-step
        Ke_einsum_3 = _integrate_stiffness_einsum_3(
            weights, dN_dxi_tensor, dN_dxi_T_tensor, D, element.detJ
        )
        # (c) Nested loop
        Ke_nested = _integrate_stiffness_nested(
            weights, dN_dxi_tensor, dN_dxi_T_tensor, D, element.detJ
        )

        # ------------------------------------------------------------------
        # (7) LOG & RETURN RESULTS
        # ------------------------------------------------------------------
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

        Ke = np.einsum("gmk,kn,gnj,g->mj", dN_dxi_T_tensor, D, dN_dxi_tensor, weights) * detJ

    Indices:
    --------
        g : Gauss point index
        m,j : the 6×6 stiffness matrix indices
        k,n : the 'component' directions (3 for Euler–Bernoulli)

    Shapes:
    -------
        dN_dxi_T_tensor : (g, 6, 3)
        D               : (3, 3)
        dN_dxi_tensor   : (g, 3, 6)
        weights         : (g,)

    Parameters
    ----------
    weights : ndarray of shape (g,)
        Gauss point weights.
    dN_dxi_tensor : ndarray of shape (g, 3, 6)
        Derivatives of shape functions w.r.t. ξ.
    dN_dxi_T_tensor : ndarray of shape (g, 6, 3)
        Transpose of dN_dxi_tensor.
    D : ndarray of shape (3, 3)
        Material stiffness matrix (for EB beam: EA, 0, EI).
    detJ : float
        Determinant of the Jacobian.

    Returns
    -------
    ndarray of shape (6,6)
        The integrated stiffness matrix.
    """
    # Single-step einsum for integration
    Ke = np.einsum("gmk,kn,gnj,g->mj", dN_dxi_T_tensor, D, dN_dxi_tensor, weights) * detJ
    return Ke


def _integrate_stiffness_einsum_3(weights, dN_dxi_tensor, dN_dxi_T_tensor, D, detJ):
    """
    Computes the stiffness matrix using a three-step einsum-based integration.

    Method:
    -------
      1) intermediate1 = dN_dxi_T_tensor × D
         shape: (g, 6, 3)
      2) intermediate2 = intermediate1 × dN_dxi_tensor
         shape: (g, 6, 6)
      3) sum over Gauss points with the weights
         shape: (6, 6)

    Parameters
    ----------
    weights : ndarray of shape (g,)
        Gauss point weights.
    dN_dxi_tensor : ndarray of shape (g, 3, 6)
        Derivatives of shape functions w.r.t. ξ.
    dN_dxi_T_tensor : ndarray of shape (g, 6, 3)
        Transpose of dN_dxi_tensor.
    D : ndarray of shape (3,3)
        Material stiffness matrix.
    detJ : float
        Determinant of the Jacobian.

    Returns
    -------
    ndarray of shape (6,6)
        The integrated stiffness matrix.
    """
    n_gauss = weights.shape[0]

    # ------------------------------
    # (1) Intermediate1 = dN_dxi_T_tensor * D
    # ------------------------------
    intermediate1 = np.einsum("gmk,kn->gmn", dN_dxi_T_tensor, D)
    log_tensor_operation(
        "Intermediate1 = dN_dxi_T * D (n, 6, 3)",
        intermediate1,
        [{"n": i+1, "xi": "", "w": ""} for i in range(n_gauss)]
    )

    # ------------------------------
    # (2) Intermediate2 = Intermediate1 * dN_dxi_tensor
    # ------------------------------
    intermediate2 = np.einsum("gmn,gnj->gmj", intermediate1, dN_dxi_tensor)
    log_tensor_operation(
        "Intermediate2 = Intermediate1 * dN_dxi (n, 6, 6)",
        intermediate2,
        [{"n": i+1, "xi": "", "w": ""} for i in range(n_gauss)]
    )

    # ------------------------------
    # (3) Integrate over Gauss points
    # ------------------------------
    stiffness_matrix = np.einsum("g,gmj->mj", weights, intermediate2) * detJ
    return stiffness_matrix


def _integrate_stiffness_nested(weights, dN_dxi_tensor, dN_dxi_T_tensor, D, detJ):
    """
    Computes the stiffness matrix using a nested-loop integration approach (fallback method).

    This method explicitly loops over Gauss points, the 6×6 indices of the stiffness matrix,
    and the 3 'component' directions. The result should match the einsum-based approaches.

    Parameters
    ----------
    weights : ndarray of shape (g,)
        Gauss point weights.
    dN_dxi_tensor : ndarray of shape (g, 3, 6)
        Derivatives of shape functions w.r.t. ξ.
    dN_dxi_T_tensor : ndarray of shape (g, 6, 3)
        Transpose of dN_dxi_tensor.
    D : ndarray of shape (3,3)
        Material stiffness matrix (e.g., for EB: diag(EA, 0, EI)).
    detJ : float
        Determinant of the Jacobian.

    Returns
    -------
    ndarray of shape (6,6)
        The integrated stiffness matrix.
    """
    n_gauss = weights.shape[0]
    stiffness_matrix = np.zeros((6, 6))

    # Nested loops over gauss points g, matrix indices i,j, and component directions a,b
    for g in range(n_gauss):
        w = weights[g]
        for i in range(6):
            for j in range(6):
                sum_term = 0.0
                for a in range(3):
                    for b in range(3):
                        sum_term += (
                            dN_dxi_T_tensor[g, i, a] *
                            D[a, b] *
                            dN_dxi_tensor[g, b, j]
                        )
                stiffness_matrix[i, j] += w * sum_term * detJ

    return stiffness_matrix
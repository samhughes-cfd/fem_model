# pre_processing/element_library/utilities/jacobian.py

import numpy as np

def compute_jacobian_matrix(shape_function_derivatives, node_coordinates):
    """
    Compute the Jacobian matrix for an element based on shape function derivatives
    and nodal coordinates.

    Args:
        shape_function_derivatives (ndarray): Derivatives of shape functions w.r.t. natural coordinates.
                                              Shape: (num_nodes, dim).
        node_coordinates (ndarray): Nodal coordinates of the element. Shape: (num_nodes, spatial_dim).

    Returns:
        ndarray: The Jacobian matrix. Shape: (spatial_dim, dim).
    """
    return np.dot(node_coordinates.T, shape_function_derivatives)

def compute_jacobian_determinant(jacobian_matrix):
    """
    Compute the determinant of the Jacobian matrix.

    Args:
        jacobian_matrix (ndarray): The Jacobian matrix. Shape: (spatial_dim, dim).

    Returns:
        float: Determinant of the Jacobian matrix.
    """
    return np.linalg.det(jacobian_matrix)

def general_jacobian_and_determinant(shape_function_derivatives, node_coordinates):
    """
    Compute both the Jacobian matrix and its determinant in one step.

    Args:
        shape_function_derivatives (ndarray): Derivatives of shape functions w.r.t. natural coordinates.
        node_coordinates (ndarray): Nodal coordinates of the element.

    Returns:
        tuple: (jacobian_matrix, jacobian_determinant)
            - jacobian_matrix (ndarray): Shape: (spatial_dim, dim).
            - jacobian_determinant (float): Determinant of the Jacobian matrix.
    """
    jacobian_matrix = compute_jacobian_matrix(shape_function_derivatives, node_coordinates)
    jacobian_determinant = compute_jacobian_determinant(jacobian_matrix)
    return jacobian_matrix, jacobian_determinant
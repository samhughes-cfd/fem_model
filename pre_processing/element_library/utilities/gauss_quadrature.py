# pre_processing\element_library\utilities\gauss_quadrature.py

import numpy as np

def get_gauss_points(n, dim=1):
    """
    Get Gauss-Legendre quadrature points and weights for 1D, 2D, or 3D integration in natural coordinates.

    Parameters:
        n (int): Number of Gauss points per dimension.
        dim (int): Dimension of integration (1, 2, or 3).

    Returns:
        tuple: (points, weights)
            - points (ndarray): Array of shape (num_points, dim) containing Gauss points in each dimension.
            - weights (ndarray): Corresponding weights for each Gauss point.
    """
    # Compute 1D Gauss points and weights
    xi_points, xi_weights = np.polynomial.legendre.leggauss(n)

    if dim == 1:
        return xi_points.reshape(-1, 1), xi_weights

    elif dim == 2:
        # Use NumPy meshgrid instead of itertools.product
        P1, P2 = np.meshgrid(xi_points, xi_points, indexing='ij')
        W1, W2 = np.meshgrid(xi_weights, xi_weights, indexing='ij')

        # Reshape into 2D arrays: (num_points**2, 2)
        points = np.column_stack((P1.ravel(), P2.ravel()))
        weights = (W1 * W2).ravel()  # Element-wise multiplication for weights

    elif dim == 3:
        # Use NumPy meshgrid instead of itertools.product
        P1, P2, P3 = np.meshgrid(xi_points, xi_points, xi_points, indexing='ij')
        W1, W2, W3 = np.meshgrid(xi_weights, xi_weights, xi_weights, indexing='ij')

        # Reshape into 3D arrays: (num_points**3, 3)
        points = np.column_stack((P1.ravel(), P2.ravel(), P3.ravel()))
        weights = (W1 * W2 * W3).ravel()  # Element-wise multiplication for weights

    else:
        raise ValueError("Dimension must be 1, 2, or 3.")

    return points, weights


def integrate_matrix(n_gauss, integrand_func, jacobian_func, dim=1):
    """
    Perform numerical integration of a matrix over the element in natural coordinates using Gauss quadrature.

    Parameters:
        n_gauss (int): Number of Gauss points per dimension.
        integrand_func (callable): Function that computes the integrand matrix at a given natural coordinate xi.
                                   Should accept an ndarray xi of shape (dim,) and return an ndarray (matrix).
        jacobian_func (callable): Function that computes the determinant of the Jacobian at xi.
                                  Should accept an ndarray xi of shape (dim,) and return a scalar.
        dim (int): Dimension of integration (1, 2, or 3).

    Returns:
        ndarray: Integrated matrix over the element.
    """
    # Get Gauss points and weights
    points, weights = get_gauss_points(n_gauss, dim=dim)

    # Vectorized integration
    integrands = np.array([integrand_func(xi) for xi in points])
    detJ_values = np.array([jacobian_func(xi) for xi in points])

    # Weighted summation
    integrated_matrix = np.tensordot(integrands, weights * detJ_values, axes=(0, 0))

    return integrated_matrix


def integrate_vector(n_gauss, integrand_func, jacobian_func, dim=1):
    """
    Perform numerical integration of a vector over the element in natural coordinates using Gauss quadrature.

    Parameters:
        n_gauss (int): Number of Gauss points per dimension.
        integrand_func (callable): Function that computes the integrand vector at a given natural coordinate xi.
                                   Should accept an ndarray xi of shape (dim,) and return an ndarray (vector).
        jacobian_func (callable): Function that computes the determinant of the Jacobian at xi.
                                  Should accept an ndarray xi of shape (dim,) and return a scalar.
        dim (int): Dimension of integration (1, 2, or 3).

    Returns:
        ndarray: Integrated vector over the element.
    """
    # Get Gauss points and weights
    points, weights = get_gauss_points(n_gauss, dim=dim)

    # Vectorized integration
    integrands = np.array([integrand_func(xi) for xi in points])
    detJ_values = np.array([jacobian_func(xi) for xi in points])

    # Weighted summation
    integrated_vector = np.tensordot(integrands, weights * detJ_values, axes=(0, 0))

    return integrated_vector
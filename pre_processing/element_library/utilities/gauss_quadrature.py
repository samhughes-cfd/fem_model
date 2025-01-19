# pre_processing\element_library\utilities\gauss_quadrature.py

import numpy as np
from itertools import product

def get_gauss_points(n, dim=1):
    """
    Get Gauss-Legendre quadrature points and weights for 1D, 2D, or 3D integration in natural coordinates.
    
    Parameters:
        n (int): Number of Gauss points per dimension.
        dim (int): Dimension of integration (1, 2, or 3).
    
    Returns:
        tuple: (points, weights)
            points (ndarray): Array of shape (num_points, dim) containing Gauss points in each dimension.
            weights (ndarray): Corresponding weights for each Gauss point.
    """
    if dim == 1:
        # 1D Gauss points and weights
        xi_points, xi_weights = np.polynomial.legendre.leggauss(n)
        points = points.reshape(-1, 1)  # Ensure points are 2D array with shape (num_points, 1)
        return points, weights
    elif dim == 2:
        # 2D Gauss points and weights (tensor product of 1D points)
        xi_points, xi_weights = np.polynomial.legendre.leggauss(n)
        eta_points, eta_weights = xi_points, xi_weights
        points = np.array(list(product(xi_points, eta_points)))
        weights = np.array([w1 * w2 for w1, w2 in product(xi_weights, eta_weights)])
        return points, weights
    elif dim == 3:
        # 3D Gauss points and weights (tensor product of 1D points)
        xi_points, xi_weights = np.polynomial.legendre.leggauss(n)
        eta_points, eta_weights = xi_points, xi_weights
        zeta_points, zeta_weights = xi_points, xi_weights
        points = np.array(list(product(xi_points, eta_points, zeta_points)))
        weights = np.array([w1 * w2 * w3 for w1, w2, w3 in product(xi_weights, eta_weights, zeta_weights)])
        return points, weights
    else:
        raise ValueError("Dimension must be 1, 2, or 3.")

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
    
    # Initialize the integrated matrix with zeros
    sample_matrix = integrand_func(points[0])
    integrated_matrix = np.zeros_like(sample_matrix)
    
    for xi, wi in zip(points, weights):
        # Compute the determinant of the Jacobian at xi
        detJ = jacobian_func(xi)
        
        # Evaluate the integrand at xi
        integrand = integrand_func(xi)
        
        # Accumulate the weighted integrand, adjusted for the Jacobian determinant
        integrated_matrix += integrand * wi * detJ
    
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
    
    # Initialize the integrated vector with zeros
    sample_vector = integrand_func(points[0])
    integrated_vector = np.zeros_like(sample_vector)
    
    for xi, wi in zip(points, weights):
        # Compute the determinant of the Jacobian at xi
        detJ = jacobian_func(xi)
        
        # Evaluate the integrand at xi
        integrand = integrand_func(xi)
        
        # Accumulate the weighted integrand, adjusted for the Jacobian determinant
        integrated_vector += integrand * wi * detJ
    
    return integrated_vector
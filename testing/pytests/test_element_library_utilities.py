"""
Tests utilities: Shape functions, Jacobian, Gauss quadrature, and DOF mapping.
"""

import testing
import os
import sys
import logging
import numpy as np

# Logging configuration
LOG_FILE = os.path.join(os.path.dirname(__file__), "pytest_utilities.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, mode="w")]
)

# Add project root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# FEM imports
from pre_processing.element_library.utilities.shape_function_library.euler_bernoulli_sf import euler_bernoulli_shape_functions
from pre_processing.element_library.utilities.jacobian import compute_jacobian_matrix, compute_jacobian_determinant
from pre_processing.element_library.utilities.gauss_quadrature import integrate_matrix
from pre_processing.element_library.utilities.dof_mapping import expand_dof_mapping

def test_shape_function_library():
    """Tests Euler-Bernoulli shape function computations."""
    xi = 0.0
    L = 8.0  
    N, dN_dxi, d2N_dxi2 = euler_bernoulli_shape_functions(xi, L)

    assert isinstance(N, np.ndarray), "Shape function N should be a numpy array"
    assert isinstance(dN_dxi, np.ndarray), "dN/dxi should be a numpy array"
    assert isinstance(d2N_dxi2, np.ndarray), "d2N/dxi2 should be a numpy array"

    logging.info(f"Shape Functions at xi=0:\nN: {N}\ndN_dxi: {dN_dxi}\nd2N_dxi2: {d2N_dxi2}")

def test_jacobian_and_determinant():
    """Tests Jacobian computations."""
    shape_derivatives = np.array([[0.5], [-0.5]])
    node_coordinates = np.array([[0], [8.0]])  

    J = compute_jacobian_matrix(shape_derivatives, node_coordinates)
    detJ = compute_jacobian_determinant(J)

    assert J.shape == (1, 1), f"Jacobian matrix should be (1,1), got {J.shape}"
    assert isinstance(detJ, float), "Jacobian determinant should be a scalar"

    logging.info(f"Jacobian Matrix:\n{J}\nDeterminant: {detJ}")

def test_gauss_quadrature():
    """Tests Gauss quadrature integration."""
    def integrand(xi):
        return np.array([[xi ** 2]])

    def jacobian_func(xi):
        return 1.0 

    integrated_matrix = integrate_matrix(n_gauss=3, integrand_func=integrand, jacobian_func=jacobian_func)

    assert integrated_matrix.shape == (1, 1), f"Integrated matrix should be (1,1), got {integrated_matrix.shape}"
    logging.info(f"Gauss Quadrature Integrated Matrix:\n{integrated_matrix}")

def test_dof_mapping():
    """Tests DOF mapping utility."""
    reduced_array = np.array([1, 2, 3])
    dof_map = [0, 2, 4]
    expanded = expand_dof_mapping(reduced_array, full_size=6, dof_map=dof_map)

    assert expanded.shape == (6,), f"Expanded array should be (6,), got {expanded.shape}"
    logging.info(f"Expanded DOF Mapped Array:\n{expanded}")
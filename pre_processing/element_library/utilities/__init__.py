# pre_processing\element_library\utilities\__init__.py

# Import shape function sets
from pre_processing.element_library.utilities.shape_function_library.timoshenko_sf import timoshenko_shape_functions
from pre_processing.element_library.utilities.shape_function_library.euler_bernoulli_sf import euler_bernoulli_shape_functions
from pre_processing.element_library.utilities.dof_mapping import expand_dof_mapping
from pre_processing.element_library.utilities.gauss_quadrature import get_gauss_points, integrate_vector, integrate_matrix
from pre_processing.element_library.utilities.jacobian import compute_jacobian_matrix, compute_jacobian_determinant, general_jacobian_and_determinant

# Define explicitly exported members
__all__ = ["timoshenko_shape_functions", "euler_bernoulli_shape_functions",
           "expand_dof_mapping", 
           "get_gauss_points", "integrate_vector", "integrate_matrix",
           "compute_jacobian_matrix", "compute_jacobian_determinant", "general_jacobian_and_determinant"]
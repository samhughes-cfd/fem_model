# pre_processing\element_library\euler_bernoulli\utilities\__init__.py

# Import the main element class
from .element_force_vector_3DOF import compute_force_vector
from .element_stiffness_matrix_3DOF import compute_stiffness_matrix
from .shape_functions_3DOF import euler_bernoulli_shape_functions

# Define explicitly exported members
__all__ = ["EulerBernoulliBeamElement3DOF",
           "compute_force_vector",
           "compute_stiffness_matrix",
           "euler_bernoulli_shape_functions"]
"""
processing\solver_registry.py

This module provides a registry of linear solvers for finite element analysis (FEM). 
It includes direct, iterative, and specialized solvers from SciPy, offering flexibility 
for solving dense, sparse, or ill-conditioned systems.

Role in Pipeline:
- Enables flexible solver selection for FEM problems.
- Centralizes solver definitions to simplify integration with other components.

"""

from scipy.sparse.linalg import cg, gmres, minres, bicg, bicgstab, lsmr, lsqr, spsolve
from scipy.linalg import solve, lu_factor, lu_solve

def get_solver_registry():
    """
    Returns a registry of SciPy solvers available for solving linear systems.

    Returns:
        dict: Mapping solver names (str) to functions.
    """
    return {
        "direct_solver_dense": solve,
        "lu_decomposition_solver": lambda A, b: lu_solve(lu_factor(A), b),
        "direct_solver_sparse": spsolve,
        "conjugate_gradient_solver": cg,
        "generalized_minimal_residual_solver": gmres,
        "minimum_residual_solver": minres,
        "bi-conjugate_gradient_solver": bicg,
        "bi-conjugate_gradient_stabilized_solver": bicgstab,
        "least_squares_solver": lsmr,
        "sparse_least_squares_solver": lsqr,
    }
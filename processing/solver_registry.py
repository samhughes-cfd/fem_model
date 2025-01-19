"""
processing\solver_registry.py

This module provides a registry of linear solvers for finite element analysis (FEM). 
It includes direct, iterative, and specialized solvers from SciPy, offering flexibility 
for solving dense, sparse, or ill-conditioned systems.

Role in Pipeline:
- Enables flexible solver selection for FEM problems.
- Centralizes solver definitions to simplify integration with other components.

"""

from scipy.sparse.linalg import cg, gmres, minres, bicg, bicgstab, lsmr, lsqr
from scipy.linalg import solve, lu_factor, lu_solve

from scipy.sparse.linalg import spsolve, cg, gmres, minres, bicg, bicgstab, lsmr, lsqr
from scipy.linalg import solve, lu_factor, lu_solve

def get_solver_registry():
    """
    Returns a registry of SciPy solvers available for solving linear systems.

    Returns:
        dict: Mapping solver names (str) to functions.
    """
    return {
        "Direct Solver (Dense)": solve,
        "LU Decomposition Solver": lambda A, b: lu_solve(lu_factor(A), b),
        "Sparse Direct Solver": spsolve,
        "Conjugate Gradient Solver": cg,
        "Generalized Minimal Residual Solver": gmres,
        "Minimum Residual Solver": minres,
        "Bi-Conjugate Gradient Solver": bicg,
        "Bi-Conjugate Gradient Stabilized Solver": bicgstab,
        "Least Squares Solver": lsmr,
        "Sparse Least Squares Solver": lsqr,
    }

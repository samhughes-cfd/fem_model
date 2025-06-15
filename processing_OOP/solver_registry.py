# processing_OOP\solver_registry.py

from scipy.sparse.linalg import cg, gmres, minres, bicg, bicgstab, lsmr, lsqr, spsolve
from scipy.linalg import solve, lu_factor, lu_solve

def conjugate_gradient_solver(A, b, M=None, callback=None, **kwargs):
    """Wrapper around scipy.sparse.linalg.cg with consistent signature and kwarg passthrough."""
    return cg(A, b, M=M, callback=callback, **kwargs)

class LinearSolverRegistry:
    """
    A centralized registry of linear solvers for FEM systems with class methods
    for solver management and retrieval.
    """

    _registry = {
        "direct_solver_dense": solve,
        "lu_decomposition_solver": lambda A, b: lu_solve(lu_factor(A), b),
        "direct_solver_sparse": spsolve,
        "conjugate_gradient_solver": conjugate_gradient_solver,
        "generalized_minimal_residual_solver": gmres,
        "minimum_residual_solver": minres,
        "bi-conjugate_gradient_solver": bicg,
        "bi-conjugate_gradient_stabilized_solver": bicgstab,
        "least_squares_solver": lsmr,
        "sparse_least_squares_solver": lsqr,
    }

    # Register aliases for user-friendly access
    _registry["cg"] = _registry["conjugate_gradient_solver"]
    _registry["gmres"] = _registry["generalized_minimal_residual_solver"]
    _registry["minres"] = _registry["minimum_residual_solver"]
    _registry["bicg"] = _registry["bi-conjugate_gradient_solver"]
    _registry["bicgstab"] = _registry["bi-conjugate_gradient_stabilized_solver"]
    _registry["lsmr"] = _registry["least_squares_solver"]
    _registry["lsqr"] = _registry["sparse_least_squares_solver"]

    @classmethod
    def get_solver_registry(cls) -> dict:
        return cls._registry.copy()

    @classmethod
    def get_solver(cls, solver_name: str):
        if solver_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Invalid solver '{solver_name}'. Available solvers: {available}"
            )
        return cls._registry[solver_name]

    @classmethod
    def list_solvers(cls) -> list:
        return sorted(cls._registry.keys())

    @classmethod
    def solver_exists(cls, solver_name: str) -> bool:
        return solver_name in cls._registry
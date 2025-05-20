# processing_OOP\solver_registry.py

from scipy.sparse.linalg import cg, gmres, minres, bicg, bicgstab, lsmr, lsqr, spsolve
from scipy.linalg import solve, lu_factor, lu_solve

class LinearSolverRegistry:
    """
    A centralized registry of linear solvers for FEM systems with class methods
    for solver management and retrieval.
    
    Maintains a static registry of solver functions that can be extended
    without modifying base class functionality.
    """
    
    _registry = {
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

    @classmethod
    def get_solver_registry(cls) -> dict:
        """Return a copy of the solver registry to prevent accidental modification."""
        return cls._registry.copy()

    @classmethod
    def get_solver(cls, solver_name: str) -> callable:
        """
        Retrieve a solver function by name with validation.
        
        Args:
            solver_name: Name of the solver to retrieve
            
        Returns:
            Registered solver function
            
        Raises:
            ValueError: If solver name not found in registry
        """
        if solver_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Invalid solver '{solver_name}'. Available solvers: {available}"
            )
        return cls._registry[solver_name]

    @classmethod
    def list_solvers(cls) -> list:
        """Return sorted list of registered solver names."""
        return sorted(cls._registry.keys())

    @classmethod
    def solver_exists(cls, solver_name: str) -> bool:
        """Check if a solver exists in the registry."""
        return solver_name in cls._registry
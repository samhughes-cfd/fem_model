import logging
import numpy as np
from processing.solver_registry import get_solver_registry

logger = logging.getLogger(__name__)

def solve_fem_system(K_mod, F_mod, solver_name):
    """
    Solves the FEM system for nodal displacements using the selected solver.

    Args:
        K_mod (scipy.sparse matrix): The modified global stiffness matrix.
        F_mod (numpy.ndarray): The modified global force vector.
        solver_name (str): The name of the solver function to use.

    Returns:
        numpy.ndarray: The computed nodal displacements.

    Raises:
        ValueError: If the solver name is not in the registry.
        RuntimeError: If the solver fails to solve the system.
    """
    logger.info(f"Solving FEM system using `{solver_name}`.")

    # Get registered solver
    solver_registry = get_solver_registry()
    if solver_name not in solver_registry:
        raise ValueError(f"Solver '{solver_name}' is not registered.")

    solver_func = solver_registry[solver_name]

    # Solve using selected solver
    try:
        return solver_func(K_mod, F_mod)
    except Exception as e:
        raise RuntimeError(f"Solver '{solver_name}' encountered an error: {str(e)}")
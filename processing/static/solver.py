# processing\static\solver.py

import os
import time
import logging
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from processing.solver_registry import get_solver_registry

logger = logging.getLogger(__name__)

def solve_fem_system(K_mod, F_mod, solver_name, job_results_dir):
    """
    Solves the FEM system for nodal displacements using the selected solver,
    logs solver-specific performance, and generates performance plots.

    Args:
        K_mod (scipy.sparse matrix or ndarray): The modified global stiffness matrix.
        F_mod (numpy.ndarray): The modified global force vector.
        solver_name (str): The name of the solver function to use.
        job_results_dir (str): Directory where solver performance logs and plots are stored.

    Returns:
        numpy.ndarray: The computed nodal displacements.

    Raises:
        ValueError: If the solver name is not in the registry.
        RuntimeError: If the solver fails to solve the system.
    """

    logger.info(f"Solving FEM system using `{solver_name}`.")

    # Ensure output directory exists
    os.makedirs(job_results_dir, exist_ok=True)
    log_filepath = os.path.join(job_results_dir, "solver_performance.log")

    # Get registered solver
    solver_registry = get_solver_registry()
    if solver_name not in solver_registry:
        raise ValueError(f"Solver '{solver_name}' is not registered.")

    solver_func = solver_registry[solver_name]

    # Check if K_mod is sparse
    is_sparse = sp.issparse(K_mod)

    with open(log_filepath, 'a', encoding="utf-8") as log_file:
        log_file.write("\n" + "-" * 60 + "\n")
        log_file.write(f"### Solver Performance: {solver_name}\n")

        # Compute condition number if feasible
        try:
            cond_number = np.linalg.cond(K_mod.toarray()) if is_sparse else np.linalg.cond(K_mod)
            log_file.write(f"🔹 Condition Number: {cond_number:.2e}\n")
            if cond_number > 1e10:
                log_file.write("⚠️  High condition number detected: Consider preconditioning.\n")
        except np.linalg.LinAlgError:
            log_file.write("⚠️  Condition number could not be computed (singular matrix)\n")
            cond_number = None  # Prevents plotting issues

        # Solve system and measure time
        start_time = time.time()
        try:
            U = solver_func(K_mod, F_mod)
            solve_time = time.time() - start_time
            log_file.write(f"✅ Solver completed in {solve_time:.6f} seconds\n")
        except Exception as e:
            solve_time = time.time() - start_time
            log_file.write(f"❌ Solver failed after {solve_time:.6f} seconds\n")
            log_file.write(f"⚠️ Error: {str(e)}\n")
            return None  # Exit early on failure

        # --- Iterative Solver Details ---
        num_iterations = None
        residual_norm = None
        residuals = []

        if hasattr(U, 'info'):  # Solvers returning iteration data
            num_iterations = U.info
        elif isinstance(U, tuple) and len(U) == 2:  # Solvers returning (U, info)
            U, num_iterations = U
        elif isinstance(U, tuple) and len(U) == 3:  # Solvers returning (U, info, residuals)
            U, num_iterations, residuals = U
            residual_norm = residuals[-1] if residuals else None

        # Log iteration count if applicable
        if num_iterations is not None:
            log_file.write(f"🔄 Iterations: {num_iterations}\n")
            if num_iterations > 1000:
                log_file.write("⚠️  Warning: Solver required many iterations. Consider preconditioning.\n")
        else:
            log_file.write("ℹ️  Solver did not report iterations (direct solver assumed)\n")

        # Log residual norm if available
        if residual_norm is not None:
            log_file.write(f"📌 Final Residual Norm: {residual_norm:.4e}\n")
            if residual_norm > 1e-6:
                log_file.write("⚠️  Residual is large, check convergence criteria.\n")

        # Log residual history for iterative solvers
        if residuals:
            log_file.write("📈 Residual History:\n")
            for i, res in enumerate(residuals[:10]):  # Limit to first 10 residuals
                log_file.write(f"   Iter {i+1}: {res:.4e}\n")
            if len(residuals) > 10:
                log_file.write(f"   ... ({len(residuals)} total iterations)\n")

        # Generate Performance Plots
        plot_solver_performance(
            solver_name, residuals, solve_time, cond_number, num_iterations, job_results_dir
        )

        # Log successful completion
        log_file.write("✅ Solver execution successful\n")

        return U  # Return computed displacements


def plot_solver_performance(solver_name, residuals, solve_time, cond_number, num_iterations, job_results_dir):
    """
    Generates and saves performance plots for solver convergence behavior.

    Args:
        solver_name (str): Name of the solver.
        residuals (list): Residual values at each iteration.
        solve_time (float): Solver execution time.
        cond_number (float): Condition number of the matrix.
        num_iterations (int or None): Iteration count (if applicable).
        job_results_dir (str): Directory to save plots.
    """

    # Residual History Plot (Only if iterative solver)
    if residuals:
        plt.figure(figsize=(6, 4))
        plt.semilogy(range(1, len(residuals) + 1), residuals, marker="o", linestyle="-", color="blue")
        plt.xlabel("Iteration")
        plt.ylabel("Residual Norm")
        plt.title(f"Residual Convergence: {solver_name}")
        plt.grid(True)
        plt.savefig(os.path.join(job_results_dir, "residual_history.png"))
        plt.close()

    # Solver Performance Summary Plot
    plt.figure(figsize=(6, 4))
    metrics = ["Solve Time (s)", "Condition Number", "Iterations"]
    values = [solve_time, cond_number if cond_number else 0, num_iterations if num_iterations else 0]
    
    plt.bar(metrics, values, color=["green", "red", "blue"])
    plt.ylabel("Value (log scale)")
    plt.yscale("log")
    plt.title(f"Solver Performance: {solver_name}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(job_results_dir, "solver_performance.png"))
    plt.close()
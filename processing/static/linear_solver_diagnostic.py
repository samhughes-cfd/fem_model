import os
import time
import numpy as np
import scipy.sparse as sp

def log_solver_performance(K_mod, F_mod, solver_name, solve_func, job_results_dir, label="Solver Performance"):
    """
    Logs solver performance metrics including runtime, iterations (if applicable), and condition number.

    Parameters:
        K_mod (scipy.sparse matrix or ndarray): The modified global stiffness matrix.
        F_mod (numpy.ndarray): The modified global force vector.
        solver_name (str): The name of the solver function being used.
        solve_func (callable): The function used to solve the system.
        job_results_dir (str): Directory where the log file is stored.
        label (str): Label for the solver log section.

    Logs:
        - Solver name
        - Matrix condition number (if computable)
        - Solver runtime
        - Number of iterations (if applicable)
        - Convergence status or warnings
    """

    # Ensure directory exists
    os.makedirs(job_results_dir, exist_ok=True)
    filepath = os.path.join(job_results_dir, "solver_performance.log")

    # Check if K_mod is sparse
    is_sparse = sp.issparse(K_mod)

    with open(filepath, 'a', encoding="utf-8") as log_file:
        log_file.write("\n" + "-" * 60 + "\n")
        log_file.write(f"### {label}: {solver_name}\n")

        # Compute condition number (if feasible)
        try:
            cond_number = np.linalg.cond(K_mod.toarray()) if is_sparse else np.linalg.cond(K_mod)
            log_file.write(f"🔹 Condition Number: {cond_number:.2e}\n")
        except np.linalg.LinAlgError:
            log_file.write("⚠️  Condition number could not be computed (singular matrix or ill-conditioned)\n")

        # Measure solver runtime
        start_time = time.time()
        try:
            result = solve_func(K_mod, F_mod)
            solve_time = time.time() - start_time
            log_file.write(f"✅ Solver completed in {solve_time:.4f} seconds\n")
        except Exception as e:
            solve_time = time.time() - start_time
            log_file.write(f"❌ Solver failed after {solve_time:.4f} seconds\n")
            log_file.write(f"⚠️ Error: {str(e)}\n")
            return None  # Exit early on failure

        # Check if solver has an 'info' attribute for iteration data
        num_iterations = None
        if hasattr(result, 'info') and isinstance(result.info, int):
            num_iterations = result.info

        # Log iteration count if applicable
        if num_iterations is not None:
            log_file.write(f"🔄 Iterations: {num_iterations}\n")
            if num_iterations > 1000:
                log_file.write("⚠️  Warning: Solver required a high number of iterations. Consider preconditioning.\n")
        else:
            log_file.write("ℹ️  Solver did not report iterations (direct solver assumed)\n")

        # Log successful completion
        log_file.write("✅ Solver execution successful\n")

        return result  # Return the solution
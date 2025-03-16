# processing\static\solver.py

import os
import time
import logging
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from processing.solver_registry import get_solver_registry

# ✅ Configure module-level logging
logger = logging.getLogger(__name__)

def configure_solver_logging(job_results_dir):
    """📜 Configures logging for solver performance, ensuring logs are stored in the results directory."""
    os.makedirs(job_results_dir, exist_ok=True)
    log_filepath = os.path.join(job_results_dir, "solver.log")

    # 📝 Remove existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # 📂 File handler (detailed logs)
    file_handler = logging.FileHandler(log_filepath, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    # 🖥️ Console handler (minimal logs)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    # 🎯 Set log level (Suppress excessive logs in terminal)
    logger.setLevel(logging.INFO)

def solve_fem_system(K_mod, F_mod, solver_name, job_results_dir):
    """
    🚀 Solves the FEM system for nodal displacements using the selected solver.

    Logs solver-specific performance and generates diagnostic plots.

    **Parameters:**
        🔹 `K_mod (csr_matrix)`: Modified global stiffness matrix.
        🔹 `F_mod (np.ndarray)`: Modified global force vector.
        🔹 `solver_name (str)`: Solver function name.
        🔹 `job_results_dir (str)`: Directory for storing logs/plots.

    **Returns:**
        ✅ Computed nodal displacements (`np.ndarray`), or `None` if solver fails.
    """

    # 📝 Step 1: Configure logging
    configure_solver_logging(job_results_dir)
    log_filepath = os.path.join(job_results_dir, "solver.log")

    logger.info(f"🔹 Solving FEM system using `{solver_name}`...")

    # 🔎 Step 2: Retrieve solver function from registry
    solver_registry = get_solver_registry()
    if solver_name not in solver_registry:
        logger.error(f"❌ Solver `{solver_name}` is not registered.")
        raise ValueError(f"Solver `{solver_name}` is not registered.")

    solver_func = solver_registry[solver_name]

    # 🏗️ Step 3: Matrix Properties Logging
    is_sparse = sp.issparse(K_mod)
    nnz = K_mod.nnz if is_sparse else np.count_nonzero(K_mod)
    sparsity_ratio = 1 - (nnz / (K_mod.shape[0] * K_mod.shape[1]))

    with open(log_filepath, 'a', encoding="utf-8") as log_file:
        log_file.write(f"\n🔍 Matrix Properties:\n")
        log_file.write(f"   - Shape: {K_mod.shape}\n")
        log_file.write(f"   - Sparsity: {sparsity_ratio:.4%}\n")
        log_file.write(f"   - Nonzero Entries: {nnz}\n")

    # 📊 Step 4: Compute Condition Number (if feasible)
    cond_number = None
    if is_sparse:
        try:
            cond_number = np.linalg.cond(K_mod.toarray())
            logger.info(f"📊 Condition Number: {cond_number:.2e}")
            with open(log_filepath, 'a', encoding="utf-8") as log_file:
                log_file.write(f"   - Condition Number: {cond_number:.2e}\n")
        except np.linalg.LinAlgError:
            logger.warning("⚠️ Condition number could not be computed (singular matrix).")

    # ⏳ Step 5: Solve System
    start_time = time.time()
    try:
        U = solver_func(K_mod, F_mod)
        solve_time = time.time() - start_time
        logger.info(f"✅ Solver completed in {solve_time:.6f} seconds.")
    except Exception as e:
        solve_time = time.time() - start_time
        logger.error(f"❌ Solver failed after {solve_time:.6f} seconds. Error: {e}")
        return None  # Return None to indicate failure

    # 📈 Step 6: Extract Solver Metadata
    num_iterations, residual_norm, residuals = extract_solver_metadata(U)

    with open(log_filepath, 'a', encoding="utf-8") as log_file:
        log_file.write(f"\n🔍 Solver Execution Summary:\n")
        log_file.write(f"   - Solver Execution Time: {solve_time:.6f} sec\n")
        if num_iterations is not None:
            log_file.write(f"   - Iterations: {num_iterations}\n")
        if residual_norm is not None:
            log_file.write(f"   - Final Residual Norm: {residual_norm:.4e}\n")

    # 📉 Step 7: Log Residual Drop Analysis (If iterative solver)
    if residuals:
        initial_residual = residuals[0]
        final_residual = residuals[-1]
        reduction_factor = final_residual / initial_residual
        log_reduction = np.log10(reduction_factor) if reduction_factor > 0 else None

        with open(log_filepath, 'a', encoding="utf-8") as log_file:
            log_file.write(f"\n🔍 Residual Convergence:\n")
            log_file.write(f"   - Initial Residual: {initial_residual:.4e}\n")
            log_file.write(f"   - Final Residual: {final_residual:.4e}\n")
            if log_reduction:
                log_file.write(f"   - Log Reduction: {log_reduction:.2f} orders of magnitude\n")

    # 📊 Step 8: Generate Solver Diagnostic Plots
    plot_solver_performance(solver_name, residuals, solve_time, cond_number, num_iterations, job_results_dir)

    logger.info("✅ Solver execution successful.")
    return U  # Return computed displacements

def extract_solver_metadata(U):
    """🔎 Extracts solver metadata like iteration count and residual norms if available."""
    num_iterations = None
    residual_norm = None
    residuals = []

    if hasattr(U, 'info'):
        num_iterations = U.info
    elif isinstance(U, tuple) and len(U) == 2:
        U, num_iterations = U
    elif isinstance(U, tuple) and len(U) == 3:
        U, num_iterations, residuals = U
        residual_norm = residuals[-1] if residuals else None

    return num_iterations, residual_norm, residuals

def plot_solver_performance(solver_name, residuals, solve_time, cond_number, num_iterations, job_results_dir):
    """
    📊 Generates and saves performance plots for solver convergence behavior.
    """

    # 📈 Residual History Plot (If iterative solver)
    if residuals:
        plt.figure(figsize=(6, 4))
        plt.semilogy(range(1, len(residuals) + 1), residuals, marker="o", linestyle="-", color="blue")
        plt.xlabel("Iteration")
        plt.ylabel("Residual Norm")
        plt.title(f"Residual Convergence: {solver_name}")
        plt.grid(True)
        plt.savefig(os.path.join(job_results_dir, "residual_history.png"))
        plt.close()

    # 📊 Solver Performance Summary Plot
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
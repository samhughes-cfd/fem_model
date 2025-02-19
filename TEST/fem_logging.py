import os
import numpy as np

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def log_solver_metrics(K, F, solution, solve_time, output_dir, filename="solver_metrics_log.txt"):
    filepath = os.path.join(output_dir, filename)
    residual_norm = np.linalg.norm(K @ solution - F)  # Compute the residual norm.
    condition_number = np.linalg.cond(K) if K.shape[0] < 500 else "Too large to compute"

    with open(filepath, 'w', encoding="utf-8") as log_file:
        log_file.write("# Solver Performance Metrics Log\n\n")
        log_file.write(f"Solver execution time: {solve_time:.6f} seconds\n")
        log_file.write(f"Residual norm: {residual_norm:.6e}\n")
        log_file.write(f"Condition number: {condition_number}\n\n")
        # If an iterative solver was used and returns iteration info, log it.
        if isinstance(solution, tuple) and len(solution) == 2:
            log_file.write(f"Solver iterations: {solution[1]}\n")

def log_system_diagnostics(K, F, fixed_dofs, output_dir, filename="system_diagnostics_log.txt", label="System"):
    """
    Logs diagnostics for any system defined by stiffness matrix K and force vector F.
    
    Parameters:
        K (ndarray): A square stiffness matrix.
        F (ndarray): A force vector whose length matches K's dimensions.
        fixed_dofs (list or ndarray): List of DOFs that are fixed.
        output_dir (str or list or os.PathLike): Directory where the log file is saved.
            If a list is provided, its elements will be joined to form a valid path.
        filename (str): Name of the log file (default "system_diagnostics_log.txt").
        label (str): A label to identify this system in the log.
        
    The diagnostics include:
        - Total DOFs.
        - Number of nonzero rows (i.e., equations).
        - A check for underconstrained or overconstrained systems.
        - A check for positive definiteness of K.
        - Lists of strictly zero rows and columns.
        
    The information is appended to the log file.
    """
    # Ensure output_dir is a proper string/path.
    if isinstance(output_dir, (list, tuple)):
        output_dir = os.path.join(*output_dir)
    
    filepath = os.path.join(output_dir, filename)
    
    # Validate dimensions.
    n = K.shape[0]
    if K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix.")
    if F.shape[0] != n:
        raise ValueError("F must have the same length as the number of rows in K.")
    
    # Count the number of equations (nonzero rows).
    num_equations = np.count_nonzero(~(K == 0).all(axis=1))
    
    with open(filepath, 'a', encoding="utf-8") as log_file:
        log_file.write(f"### Diagnostics for {label}\n")
        log_file.write(f"Total DOFs: {n}\n")
        log_file.write(f"Total Equations (nonzero rows in K): {num_equations}\n")
        log_file.write(f"Fixed DOFs: {fixed_dofs}\n")
        
        if num_equations < n:
            log_file.write("⚠️ Underconstrained system detected\n")
        elif num_equations > n:
            log_file.write("⚠️ Overconstrained system detected\n")
        else:
            log_file.write("✅ Well-posed system detected\n")
        
        # Check positive definiteness using Cholesky factorization.
        try:
            np.linalg.cholesky(K)
            log_file.write("✅ K is positive definite (Invertible)\n")
        except np.linalg.LinAlgError:
            log_file.write("⚠️ Singular matrix detected! K is not invertible\n")
        
        # Identify zero rows and columns.
        zero_rows = list(np.where(~K.any(axis=1))[0])
        zero_cols = list(np.where(~K.any(axis=0))[0])
        if zero_rows:
            log_file.write(f"⚠️ Zero rows found at indices: {zero_rows}\n")
        if zero_cols:
            log_file.write(f"⚠️ Zero columns found at indices: {zero_cols}\n")
        log_file.write("\n")
# simulation_runner\static\linear_static_diagnostic.py

import os
import numpy as np
import scipy.sparse as sp

def log_system_diagnostics(K, F, bc_dofs=None, job_results_dir="", filename="system_diagnostics.log", label="System"):
    """
    Logs diagnostics for a system defined by stiffness matrix K and force vector F.

    Parameters:
        K (ndarray or sparse matrix): Square stiffness matrix (should not be modified).
        F (ndarray): Force vector matching K's row count (should not be modified).
        bc_dofs (list or ndarray, optional): List of DOFs with applied boundary conditions.
        job_results_dir (str): Directory where logs are stored.
        filename (str): Log filename (default "system_diagnostics.log").
        label (str): Label for the system in the log.

    Logs:
        - Total DOFs
        - Number of independent equations (nonzero rows)
        - Under/over constraint warnings
        - Boundary condition DOFs
        - Matrix singularity check
        - Positive definiteness check
        - Matrix condition number (if applicable)
        - Zero row/column checks
        - **Force Vector Analysis**
            - Zero forces
            - Min/max forces per DOF
            - Summation check (should be ~0 for equilibrium)
            - Forces on constrained DOFs

    Does **not** modify `K` or `F`.
    """

    # Ensure directory exists
    os.makedirs(job_results_dir, exist_ok=True)
    filepath = os.path.join(job_results_dir, filename)

    # Validate dimensions
    n = K.shape[0]
    if K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix.")
    if F.shape[0] != n:
        raise ValueError("F must match K's dimensions.")

    # Check if K is sparse
    is_sparse = sp.issparse(K)

    # Count nonzero rows (active equations)
    num_equations = np.count_nonzero(K.getnnz(axis=1) > 0) if is_sparse else np.count_nonzero(~(K == 0).all(axis=1))

    # Ensure bc_dofs is a list or None
    if bc_dofs is None or (isinstance(bc_dofs, np.ndarray) and bc_dofs.size == 0):
        bc_dofs = list(map(int, np.where(K.diagonal() > 1e11)[0])) if is_sparse else list(map(int, np.where(np.diag(K) > 1e11)[0]))
    else:
        bc_dofs = bc_dofs.tolist() if isinstance(bc_dofs, np.ndarray) else bc_dofs

    with open(filepath, 'a', encoding="utf-8") as log_file:
        log_file.write("\n" + "-" * 60 + "\n")
        log_file.write(f"### Diagnostics for {label}\n")
        log_file.write(f"Total DOFs: {n}\n")
        log_file.write(f"Independent Equations (nonzero rows in K): {num_equations}\n")
        log_file.write(f"Boundary Condition DOFs: {bc_dofs if len(bc_dofs) > 0 else 'None detected'}\n")

        # Constraint Status
        if num_equations < n:
            log_file.write("⚠️  Underconstrained system detected (too few equations)\n")
        elif num_equations > n:
            log_file.write("⚠️  Overconstrained system detected (too many equations)\n")
        else:
            log_file.write("✅  Well-posed system detected\n")

        # Check for positive definiteness (Cholesky factorization)
        try:
            if is_sparse:
                np.linalg.cholesky(K.toarray())  
            else:
                np.linalg.cholesky(K)
            log_file.write("✅  K is positive definite (Invertible)\n")
        except np.linalg.LinAlgError:
            log_file.write("⚠️  Singular matrix detected! K is not invertible\n")

        # Compute condition number (if applicable)
        try:
            cond_number = np.linalg.cond(K.toarray()) if is_sparse else np.linalg.cond(K)
            log_file.write(f"🔹 Condition Number: {cond_number:.2e}\n")
        except np.linalg.LinAlgError:
            log_file.write("⚠️  Condition number could not be computed (singular matrix)\n")

        # Zero row and column detection
        zero_rows = list(map(int, np.where(K.getnnz(axis=1) == 0)[0])) if is_sparse else list(map(int, np.where(~K.any(axis=1))[0]))
        zero_cols = list(map(int, np.where(K.getnnz(axis=0) == 0)[0])) if is_sparse else list(map(int, np.where(~K.any(axis=0))[0]))

        if zero_rows:
            log_file.write(f"⚠️  Zero rows found at indices: {zero_rows}\n")
        if zero_cols:
            log_file.write(f"⚠️  Zero columns found at indices: {zero_cols}\n")

        # 🔍 **Force Vector Diagnostics**
        log_file.write("\n🔹 **Force Vector Diagnostics**\n")

        # Zero force detection
        zero_force_dofs = list(map(int, np.where(F == 0)[0]))
        if zero_force_dofs:
            log_file.write(f"⚠️  DOFs with zero applied force: {zero_force_dofs[:10]}{' ...' if len(zero_force_dofs) > 10 else ''}\n")

        # Min/max force values
        min_force = np.min(F)
        max_force = np.max(F)
        log_file.write(f"📌 Min Force: {min_force:.4e}, Max Force: {max_force:.4e}\n")

        # Summation check (should be ~0 in static equilibrium)
        force_sum = np.sum(F)
        log_file.write(f"📌 Sum of all forces: {force_sum:.4e} {'⚠️ (Nonzero: Possible imbalance!)' if abs(force_sum) > 1e-6 else '✅ (Near zero: Equilibrium)'}\n")

        # Check forces on BC DOFs (should be zero)
        if len(bc_dofs) > 0:
            force_at_bc_dofs = np.abs(F[bc_dofs])
            if np.any(force_at_bc_dofs > 1e-6):
                log_file.write(f"⚠️  Nonzero forces detected at fixed DOFs! Values: {force_at_bc_dofs}\n")
            else:
                log_file.write("✅  No forces detected at constrained DOFs (Correct).\n")

        # 🔹 **Force & Moment Breakdown by DOF**
        dof_names = ["F_x", "F_y", "F_z", "M_x", "M_y", "M_z"]

        for i, dof_name in enumerate(dof_names):
            forces = F[i::6]  # Extract values for each DOF type
            min_val, max_val = np.min(forces), np.max(forces)
            log_file.write(f"🔹 {dof_name}: Min {min_val:.4e}, Max {max_val:.4e}\n")

        log_file.write("\n")
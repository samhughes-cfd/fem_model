# simulation_runner\static\linear_static_diagnostic.py

import os
import numpy as np
import scipy.sparse as sp

def log_system_diagnostics(K, F, bc_dofs=None, job_results_dir="",
                           filename="system_diagnostics.log", label="System"):
    os.makedirs(job_results_dir, exist_ok=True)
    filepath = os.path.join(job_results_dir, filename)

    n = K.shape[0]
    if K.shape[0] != K.shape[1]:
        raise ValueError("K must be square.")
    if F.shape[0] != n:
        raise ValueError("F shape mismatch with K.")

    is_sparse = sp.issparse(K)
    with open(filepath, 'a', encoding="utf-8") as log_file:
        log_file.write("\n" + "-" * 60 + "\n")
        log_file.write(f"### Diagnostics for {label}\n")

        # 1) Basic logging
        log_file.write(f"Total DOFs: {n}\n")

        # 2) Nonzero rows and columns
        if is_sparse:
            num_equations = np.count_nonzero(K.getnnz(axis=1) > 0)
        else:
            num_equations = np.count_nonzero(~(K == 0).all(axis=1))
        log_file.write(f"Independent Equations (nonzero rows): {num_equations}\n")

        # 3) Boundary conditions
        if isinstance(bc_dofs, np.ndarray):
            bc_dofs = bc_dofs.tolist()

        if bc_dofs is None or len(bc_dofs) == 0:
                log_file.write("Boundary Condition DOFs: None\n")
        else:
            log_file.write(f"Boundary Condition DOFs: {bc_dofs}\n")

        # 4) Symmetry check (for real K only)
        if not np.iscomplexobj(K):
            sym_res = (np.linalg.norm(K - K.T, ord='fro')
                       if not is_sparse else (K - K.T).count_nonzero())
            if sym_res == 0:
                log_file.write("✅ Matrix is symmetric.\n")
            else:
                log_file.write(f"⚠️ Matrix is NOT symmetric! Residual: {sym_res}\n")

        # 5) Positive definiteness / invertibility check
        try:
            if is_sparse:
                np.linalg.cholesky(K.toarray())
            else:
                np.linalg.cholesky(K)
            log_file.write("✅ K is positive definite.\n")
        except np.linalg.LinAlgError:
            log_file.write("⚠️ K is not positive definite. Possibly singular or indefinite.\n")

        # 6) Approximate rank (for moderate n)
        try:
            # If n is large, consider a truncated SVD approach
            mat = K.toarray() if is_sparse else K
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            tol = 1e-12 * s[0]
            eff_rank = np.sum(s > tol)
            log_file.write(f"🔹 Approx. rank of K: {eff_rank}/{n}\n")
            # Check for indefinite behavior: negative or zero singular values (rare but possible).
        except Exception as e:
            log_file.write(f"⚠️ Rank check not performed: {e}\n")

        # 7) Condition number
        try:
            cond_number = np.linalg.cond(K.toarray() if is_sparse else K)
            log_file.write(f"🔹 Condition Number: {cond_number:.2e}\n")
        except np.linalg.LinAlgError:
            log_file.write("⚠️ Condition number could not be computed (singular).\n")

        # 8) Zero rows/cols (again, good cross-check for BC DOFs)
        zero_rows = []
        zero_cols = []
        if is_sparse:
            zero_rows = list(map(int, np.where(K.getnnz(axis=1) == 0)[0]))
            zero_cols = list(map(int, np.where(K.getnnz(axis=0) == 0)[0]))
        else:
            zero_rows = list(map(int, np.where(~K.any(axis=1))[0]))
            zero_cols = list(map(int, np.where(~K.any(axis=0))[0]))
        if zero_rows:
            log_file.write(f"⚠️ Zero rows: {zero_rows}\n")
        if zero_cols:
            log_file.write(f"⚠️ Zero columns: {zero_cols}\n")

        # 9) Force Vector analysis
        zero_force_dofs = list(map(int, np.where(F == 0)[0]))
        if zero_force_dofs:
            log_file.write(f"⚠️ DOFs with zero force: {zero_force_dofs[:10]}{'...' if len(zero_force_dofs)>10 else ''}\n")
        fmin, fmax = np.min(F), np.max(F)
        log_file.write(f"Min Force = {fmin:.4e}, Max Force = {fmax:.4e}\n")
        fsum = np.sum(F)
        log_file.write(f"Sum of Forces = {fsum:.4e} {'(⚠️ Nonzero!)' if abs(fsum)>1e-6 else '(OK)'}\n")

        # 10) Check largest forces
        abs_forces = np.abs(F)
        sorted_dofs = np.argsort(-abs_forces)
        top_n = 10
        log_file.write("Top DOFs by force magnitude:\n")
        for i in sorted_dofs[:top_n]:
            log_file.write(f"  DOF {i}, Force = {F[i]:.4e}\n")

        # ...
        log_file.write("\n")
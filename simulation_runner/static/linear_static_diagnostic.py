# simulation_runner\static\linear_static_diagnostic.py

import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Sequence, Union


class DiagnoseLinearSystem:
    """
    Generic diagnostic logger for a linear system  A · x = b.

    Parameters
    ----------
    A : (n, n) array_like or sparse matrix
        System / coefficient matrix.
    b : (n,) array_like
        Right-hand-side vector.
    constraints, constraint_rows, bc_dofs : Sequence[int] or ndarray, optional
        Indices whose unknowns are *prescribed* (Dirichlet / essential
        conditions).  `constraints` is the preferred keyword; the other two
        remain as aliases for backward compatibility.
    job_results_dir : str or Path
        Folder where the log file is written.
    filename : str, default="system_diagnostics.log"
        Name of the log file.
    label : str, default="Linear System"
        Friendly label written into the header of the log.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        A: Union[np.ndarray, sp.spmatrix],
        b: np.ndarray,
        *,
        constraints: Sequence[int] | np.ndarray | None = None,
        constraint_rows: Sequence[int] | np.ndarray | None = None,
        bc_dofs: Sequence[int] | np.ndarray | None = None,   # ← legacy alias
        job_results_dir: str | Path = ".",
        filename: str = "system_diagnostics.log",
        label: str = "Linear System",
    ):
        # --- resolve aliases ------------------------------------------------
        if constraints is None:
            constraints = constraint_rows if constraint_rows is not None else bc_dofs
        self.constraints = np.asarray(constraints, dtype=int) if constraints is not None else np.empty(0, int)

        # --- data -----------------------------------------------------------
        self.A  = A
        self.b  = np.asarray(b, dtype=float).ravel()
        self.n  = self.A.shape[0]
        if self.b.size != self.n:
            raise ValueError("b must have length equal to A.shape[0]")

        # --- housekeeping ---------------------------------------------------
        self.out_dir  = Path(job_results_dir)
        self.filename = filename
        self.label    = label
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- write
    def write(self) -> Path:
        """Run diagnostics and append them to the log file.  Returns the path."""
        path = self.out_dir / self.filename
        is_sp = sp.issparse(self.A)

        with path.open("a", encoding="utf-8") as f:
            f.write("\n" + "-" * 60 + f"\n### Diagnostics for {self.label}\n")
            f.write(f"Size: {self.n} × {self.n}\n")
            f.write(f"Non-zeros: {self.A.nnz if is_sp else np.count_nonzero(self.A)}\n")

            # --- rank / symmetry / condition number ------------------------
            try:
                mat = self.A.toarray() if is_sp else self.A
                symmetric = np.allclose(mat, mat.T)
                f.write(f"Symmetric: {symmetric}\n")
                s = np.linalg.svd(mat, compute_uv=False)
                tol = 1e-12 * s[0]
                rank = (s > tol).sum()
                f.write(f"Approx. rank: {rank}/{self.n}\n")
                cond = np.linalg.cond(mat)
                f.write(f"Condition number: {cond:.2e}\n")
            except Exception as exc:
                f.write(f"⚠  Rank / cond-no diagnostic failed: {exc}\n")

            # --- zero rows / columns --------------------------------------
            if is_sp:
                zero_rows = np.where(self.A.getnnz(axis=1) == 0)[0]
                zero_cols = np.where(self.A.getnnz(axis=0) == 0)[0]
            else:
                zero_rows = np.where(~self.A.any(axis=1))[0]
                zero_cols = np.where(~self.A.any(axis=0))[0]
            if zero_rows.size:
                f.write(f"Zero rows: {zero_rows.tolist()}\n")
            if zero_cols.size:
                f.write(f"Zero cols: {zero_cols.tolist()}\n")

            # --- RHS stats --------------------------------------------------
            fmin, fmax = self.b.min(), self.b.max()
            f.write(f"b-vector → min {fmin:.3e} , max {fmax:.3e}\n")
            if self.b.ptp() == 0:
                f.write("⚠  b appears to be all equal values\n")

            # --- constraints -----------------------------------------------
            if self.constraints.size:
                f.write(f"Prescribed DOFs (constraints): {self.constraints.tolist()}\n")
            else:
                f.write("Prescribed DOFs: none\n")

        return path
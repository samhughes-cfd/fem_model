# processing_OOP/static/operations/solver.py

"""
processing_OOP.static.operations.solver
---------------------------------------

Robust solution of the condensed FEM system

        K_cond · U_cond = F_cond

with optional symmetric scaling, adaptive pre-conditioning, iterative /
direct fall-backs, detailed diagnostics and CSV exports.

Environment “knobs”
-------------------
FEM_DISABLE_SCALING      – any value → skip Jacobi row/col scaling
FEM_ILU_NNZ_LIMIT        – max nnz for which ILU is attempted   (default 5e4)
FEM_ILU_DROP_TOL         – ILU drop tolerance                   (default 1e-6)
FEM_ILU_FILL             – ILU fill-factor                      (default 1.0)
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Dict, Callable, Optional, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from  scipy.sparse.linalg import LinearOperator


# ───────────────────────── env defaults ──────────────────────────────
_DEF_ILU_DROP  = float(os.getenv("FEM_ILU_DROP_TOL",  "1e-6"))
_DEF_ILU_FILL  = float(os.getenv("FEM_ILU_FILL",      "1.0"))
_ILU_NNZ_LIMIT = int(float(os.getenv("FEM_ILU_NNZ_LIMIT", "5e4")))      # try ILU only if nnz < limit


# ───────────────────────── helpers ───────────────────────────────────
def _cg_with_compat(A, b, **kw):
    """
    SciPy-version-proof wrapper for `spla.cg`.
    SciPy ≤1.11 uses `tol`, ≥1.12 renamed it to `rtol`.
    """
    rtol = kw.pop("tol", None)
    if rtol is not None and "rtol" not in kw:
        kw["rtol"] = rtol
    try:                                   # SciPy ≥ 1.12
        return spla.cg(A, b, **kw)
    except TypeError as err:               # SciPy ≤ 1.11
        if "rtol" not in str(err):
            raise
        kw["tol"] = kw.pop("rtol")
        return spla.cg(A, b, **kw)


def _row_col_scale(A: sp.spmatrix, F: np.ndarray):
    """Symmetric Jacobi row/column scaling (Â = D·A·D, F̂ = D·F)."""
    d = np.abs(A.diagonal())
    d[d < 1e-14] = 1.0
    s = 1.0 / np.sqrt(d)
    D = sp.diags(s)
    return D @ A @ D, D @ F, s


def _build_ilu(A: sp.spmatrix, lg: logging.Logger) -> LinearOperator:
    """Zero-fill ILU with drop/fill taken from the environment."""
    lg.debug(f"Building ILU (drop_tol={_DEF_ILU_DROP}, fill_factor={_DEF_ILU_FILL})")
    ilu = spla.spilu(A.tocsc(),
                     drop_tol   = _DEF_ILU_DROP,
                     fill_factor= _DEF_ILU_FILL)
    fill_ratio = (ilu.L.nnz + ilu.U.nnz) / A.nnz
    lg.debug(f"ILU fill-ratio = {fill_ratio:.2f}")
    return LinearOperator(A.shape, ilu.solve)


# ───────────────────────── main class ────────────────────────────────
class SolveCondensedSystem:
    """
    Solve `K_cond · U_cond = F_cond`.

    * optional symmetric scaling (Jacobi)
    * adaptive pre-conditioners: Jacobi or ILU (auto-selected)
    * iterative solvers registered in ``processing_OOP.solver_registry``
    * fall-back to direct SuperLU
    * extensive diagnostics + CSV exports
    """

    _PRECONDITIONER_BUILDERS: Dict[str, Callable[[sp.csr_matrix, logging.Logger],
                                                 LinearOperator]] = {
        "jacobi": lambda A, _: LinearOperator(
            A.shape, dtype=A.dtype,
            matvec=lambda x: np.where(np.abs(A.diagonal()) > 1e-14,
                                      x / A.diagonal(), 0.0)),
        "ilu": _build_ilu
    }

    # ───────── construction ──────────────────────────────────────────
    def __init__(
        self,
        K_cond: sp.spmatrix,
        F_cond: np.ndarray,
        solver_name: str,
        job_results_dir: str,
        *,
        preconditioner: str | None = "auto",     # "jacobi" | "ilu" | "auto" | None
        max_mem_gb: float = 10.0
    ):
        # store data
        self.K_cond = K_cond.tocsr().astype(np.float64)
        self.F_cond = F_cond.astype(np.float64).ravel()
        self.solver_name  = solver_name.lower()
        self.preconditioner = preconditioner
        self.max_mem = max_mem_gb * 1e9

        # dirs & logging
        self.job_results_dir = Path(job_results_dir)
        self.job_results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._init_logging()

        # diagnostics store
        self.diagnostics: Dict[str, Any] = {
            "solve_phase": {}, "residuals": [], "condition_estimate": None}
        self.U_cond: Optional[np.ndarray] = None

        # size checks + optional scaling
        self._validate_sizes()
        self._scale_vec: Optional[np.ndarray] = None
        if os.getenv("FEM_DISABLE_SCALING") is None:
            self.K_cond, self.F_cond, self._scale_vec = _row_col_scale(
                self.K_cond, self.F_cond)
            self.logger.debug("Row/column scaling applied.")

        # quick singular-diagonal guard
        if np.any(np.abs(self.K_cond.diagonal()) < 1e-14):
            raise ValueError("Condensed stiffness matrix is singular "
                             "(zero detected on diagonal).  "
                             "Check boundary conditions / condensation.")

        # import solver registry lazily (user can patch their own)
        from processing_OOP.solver_registry import LinearSolverRegistry
        self._solver_registry = LinearSolverRegistry.get_solver_registry()

        if self.preconditioner:
            self._solver_registry = self._apply_preconditioner(self._solver_registry)

    # ───────── public API ────────────────────────────────────────────
    def solve(self) -> np.ndarray | None:
        self.logger.info(
            f"Solver parameters • solver='{self.solver_name}', "
            f"prec='{self.preconditioner}', "
            f"K nnz={self.K_cond.nnz}, dofs={self.K_cond.shape[0]}")

        self.diagnostics["condition_estimate"] = self._estimate_condition()
        self.logger.info(f"Condensed condition estimate: "
                         f"{self.diagnostics['condition_estimate']:.2e}")

        # choose direct / iterative path
        self.U_cond = (self._solve_direct()
                       if "direct" in self.solver_name
                       else self._solve_iterative_with_fallback())

        if self.U_cond is None:
            return None

        # un-scale back to physical units
        if self._scale_vec is not None:
            self.U_cond = self._scale_vec * self.U_cond

        # final residual
        res_norm = np.linalg.norm(self.K_cond @ self.U_cond - self.F_cond)
        self.logger.info(f"Condensed residual norm: {res_norm:.3e}")
        if res_norm > 1e-6:
            self.logger.warning("Residual is high – verify model!")

        # exports & report
        self._export_U_cond()
        self._write_report()
        return self.U_cond

    # ───────── validation / logging ──────────────────────────────────
    def _validate_sizes(self):
        if self.K_cond.shape[0] != self.F_cond.size:
            raise ValueError("K_cond and F_cond dimension mismatch")
        est_bytes = (self.K_cond.data.nbytes +
                     self.K_cond.indices.nbytes +
                     self.K_cond.indptr.nbytes +
                     self.F_cond.nbytes)
        if est_bytes > self.max_mem:
            raise MemoryError("Condensed system exceeds memory limit")

    def _init_logging(self) -> logging.Logger:
        lg = logging.getLogger(f"SolveCondensedSystem.{id(self)}")
        lg.handlers.clear()
        lg.setLevel(logging.DEBUG)
        lg.propagate = False

        log_dir  = self.job_results_dir.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "SolveCondensedSystem.log"

        try:
            fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s "
                "(Module: %(module)s, Line: %(lineno)d)"))
            lg.addHandler(fh)
        except Exception as e:
            print(f"⚠️  Cannot create solver log file: {e}")

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        lg.addHandler(sh)

        lg.debug(f"📁 Log file: {log_path}")
        return lg

    # ───────── preconditioner wrapper ────────────────────────────────
    def _apply_preconditioner(self,
                              registry: Dict[str, Callable]
                              ) -> Dict[str, Callable]:
        name = self.preconditioner or "jacobi"
        if name == "auto":
            name = "ilu" if self.K_cond.nnz < _ILU_NNZ_LIMIT else "jacobi"

        builder = self._PRECONDITIONER_BUILDERS.get(name)
        if builder is None:
            self.logger.warning(f"Unknown preconditioner '{name}', "
                                "continuing without.")
            self.preconditioner = None
            return registry

        try:
            M = builder(self.K_cond, self.logger)
        except RuntimeError as err:
            # often “factor is exactly singular” from spilu
            self.logger.warning(f"Preconditioner '{name}' failed "
                                f"({err}); proceeding without it.")
            self.preconditioner = None
            return registry

        # wrap each solver so it always receives M
        return {n: (lambda fn:
                    (lambda A, b, callback=None:
                        fn(A, b, M=M, callback=callback)))(fn)
                for n, fn in registry.items()}

    # ───────── condition estimate (rough) ────────────────────────────
    def _estimate_condition(self) -> float:
        try:
            x = np.random.rand(self.K_cond.shape[0])
            for _ in range(8):                      # power iteration
                x = self.K_cond @ x
                x /= np.linalg.norm(x)
            normA = np.linalg.norm(self.K_cond @ x)
            y, info = _cg_with_compat(self.K_cond, x, rtol=1e-2, maxiter=30)
            return np.inf if info else normA * np.linalg.norm(y)
        except Exception as exc:
            self.logger.debug(f"Condition estimate failed: {exc}")
            return np.inf

    # ───────── direct solve (SuperLU) ────────────────────────────────
    def _solve_direct(self) -> np.ndarray:
        self.logger.debug("Using SuperLU direct solver")
        t0 = time.perf_counter()
        lu = spla.splu(self.K_cond)
        self.diagnostics["solve_phase"]["factorization"] = time.perf_counter() - t0
        U = lu.solve(self.F_cond)
        self.diagnostics["solve_phase"]["solution"] = time.perf_counter() - t0
        return U

    # ───────── iterative + fall-backs ────────────────────────────────
    def _solve_iterative_with_fallback(self) -> Optional[np.ndarray]:
        if self.solver_name not in self._solver_registry:
            self.logger.error(f"No such solver '{self.solver_name}' in registry")
            return None

        solver_fn = self._solver_registry[self.solver_name]
        res_list  = self.diagnostics["residuals"]

        def cb(xk):
            r = np.linalg.norm(self.K_cond @ xk - self.F_cond)
            res_list.append(r)
            n = len(res_list)
            if n % 10 == 0:
                self.logger.debug(f"Iter {n:4d} | residual {r:.3e}")
            if n > 50 and r > 100 * res_list[0]:
                raise RuntimeError("Divergence detected")

        # ── first attempt
        try:
            t0 = time.perf_counter()
            U, info = solver_fn(self.K_cond, self.F_cond, callback=cb)
            self.diagnostics["solve_phase"]["iterative"] = time.perf_counter() - t0
            if info == 0:
                return U
            self.logger.warning(f"Solver info={info} (not converged).")
        except Exception as err:
            self.logger.warning(f"Iterative solver aborted: {err}")

        # ── second attempt with ILU (if not tried and size allows)
        if self.preconditioner not in ("ilu", None) and \
           self.K_cond.nnz < _ILU_NNZ_LIMIT:
            self.logger.info("Retrying with ILU preconditioner …")
            self.preconditioner = "ilu"
            from processing_OOP.solver_registry import LinearSolverRegistry
            self._solver_registry = self._apply_preconditioner(
                LinearSolverRegistry.get_solver_registry())
            return self._solve_iterative_with_fallback()

        # ── final fall-back to direct
        self.logger.info("Switching to direct SuperLU fall-back.")
        return self._solve_direct()

    # ───────── CSV export of U_cond (07) ─────────────────────────────
    def _export_U_cond(self):
        if self.U_cond is None:
            return
        path = self.job_results_dir / "07_U_cond.csv"
        #pd.DataFrame({"Condensed DOF": np.arange(self.U_cond.size, dtype=int),
                      #"U Value": self.U_cond}).to_csv(
            #path, index=False, float_format="%.17e")
        #self.logger.info(f"💾 Condensed displacement saved → {path}")

    # ───────── summary report (log only) ────────────────────────────
    def _write_report(self):
        rep = {
            "dofs"          : self.K_cond.shape[0],
            "nnz"           : self.K_cond.nnz,
            "density"       : self.K_cond.nnz / self.K_cond.shape[0] ** 2,
            "condition_est" : self.diagnostics["condition_estimate"],
            "solve_phases"  : self.diagnostics["solve_phase"],
            "iters"         : len(self.diagnostics["residuals"]),
            "final_residual": (self.diagnostics["residuals"][-1]
                               if self.diagnostics["residuals"] else None)
        }
        self.logger.info("📊 Condensed solver report:")
        for k, v in rep.items():
            self.logger.info(f" • {k}: {v}")
# processing_OOP/static/primary_results/compute_primary_results.py
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from dataclasses import dataclass
from typing import List, Sequence
import logging, os


@dataclass
class PrimaryResultSet:
    """Minimal container for the quantities that other pipelines reuse."""
    # ────────────────────────────────────────────── global level ───────────────
    K_global:  sp.csr_matrix        # *assembled* stiffness matrix
    F_global:  np.ndarray           # global load vector   (shape = n_dof,)
    U_global:  np.ndarray           # solved displacements (shape = n_dof,)
    R_global:  np.ndarray           # support reactions    (shape = n_dof,)
    # ────────────────────────────────────────────── BC / condensed level ──────
    K_mod:     sp.csr_matrix        # stiffness *after* BCs (penalty rows)
    F_mod:     np.ndarray           # load   *after* BCs (rows zeroed)
    K_cond:    sp.csr_matrix | None # condensed stiffness  (may be None)
    F_cond:    np.ndarray  | None   # condensed load       (may be None)
    U_cond:    np.ndarray  | None   # condensed solution   (may be None)
    # ────────────────────────────────────────────── element level ─────────────
    local_global_dof_map: List[np.ndarray]  # local DOF → global DOF map
    fixed_dofs: np.ndarray                  # fixed global DOFs map

class ComputePrimaryResults:
    """
    Gather the first-order outputs of the linear static analysis in one place.

    *Reactions are evaluated with **K_global**, **NOT** with K_mod.*

    The penalty rows/columns that `ModifyGlobalSystem` inserts into **K_mod**
    enforce the displacement B.C.s numerically, but they obliterate the real
    equilibrium in those rows (all off-diagonals are zeroed and an enormous
    diagonal value is inserted, while the corresponding rows of **F_mod** are
    also zeroed).  
    Using K_mod would therefore give either *zero* or *nonsense* forces at
    supports.  To obtain physically correct reactions we must use the pristine
    assembled matrix:

    ```text
    R_global =  K_global · U_global  –  F_global
    ```
    """

    # --------------------------------------------------------------------- init
    def __init__(
        self,
        *,
        K_global: sp.csr_matrix,
        F_global: np.ndarray,
        K_mod:    sp.csr_matrix,
        F_mod:    np.ndarray,
        K_cond:   sp.csr_matrix | None,
        F_cond:   np.ndarray    | None,
        U_cond:   np.ndarray    | None,
        U_global: np.ndarray,
        local_global_dof_map: Sequence[np.ndarray],
        fixed_dofs: np.ndarray,
        job_results_dir: str | Path | None = None,
    ):
        self.K_global  = K_global
        self.F_global  = F_global
        self.K_mod     = K_mod
        self.F_mod     = F_mod
        self.K_cond    = K_cond
        self.F_cond    = F_cond
        self.U_cond    = U_cond
        self.U_global  = U_global
        self.local_global_dof_map = list(local_global_dof_map)
        self.fixed_dofs = np.asarray(fixed_dofs, dtype=np.int32)
        self.job_results_dir  = Path(job_results_dir) if job_results_dir else None

        self.logger = self._init_logging()

    # ----------------------------------------------------------------- logging
    def _init_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"ComputePrimaryResults.{id(self)}")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if self.job_results_dir:
            logs_dir = self.job_results_dir.parent / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            f = logging.FileHandler(logs_dir / "ComputePrimaryResults.log", mode="w", encoding="utf-8")
            f.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(f)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(sh)
        return logger

    # --------------------------------------------------------------- interface
    def compute(self) -> PrimaryResultSet:
        self._basic_sanity_checks()

        # ── raw reactions for every DOF ─────────────────────────────────────
        R_raw = self.K_global @ self.U_global - self.F_global

        # ── mask free DOFs (keep zeros elsewhere) ───────────────────────────
        R_global = np.zeros_like(R_raw)
        R_global[self.fixed_dofs] = R_raw[self.fixed_dofs]

        self.logger.info("✅ Reactions evaluated & masked to fixed DOFs")

        result = PrimaryResultSet(
            K_global = self.K_global,
            F_global = self.F_global,
            K_mod    = self.K_mod,
            F_mod    = self.F_mod,
            K_cond   = self.K_cond,
            F_cond   = self.F_cond,
            U_cond   = self.U_cond,
            U_global = self.U_global,
            R_global = R_global,
            local_global_dof_map = self.local_global_dof_map,
            fixed_dofs = self.fixed_dofs,
        )
        return result

    # -------------------------------------------------------------- validators
    def _basic_sanity_checks(self) -> None:
        n = self.K_global.shape[0]
        if self.U_global.shape != (n,):
            raise ValueError(f"U_global shape {self.U_global.shape} does not match K_global ({n},)")
        if self.F_global.shape != (n,):
            raise ValueError(f"F_global shape {self.F_global.shape} does not match K_global ({n},)")
        if self.K_mod.shape != self.K_global.shape:
            raise ValueError("K_mod size mismatch with K_global")
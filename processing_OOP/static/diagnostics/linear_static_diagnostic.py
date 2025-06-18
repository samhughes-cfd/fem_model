"""
CURRENT │ EVOLUTION diagnostics for the three linear-static stages
( global , modified , condensed ).  One instance → one block in the log.
"""

from pathlib import Path
from typing  import Sequence, Union
import logging

import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse


# ── pretty two-column writer ────────────────────────────────────────────────
def _twocol(fh, key, left, right, key_w=18, col_w=18, sep=" │ "):
    fh.write(f"{key:<{key_w}}{sep}{left:<{col_w}}{sep}{right:<{col_w}}\n")


# ── main class ──────────────────────────────────────────────────────────────
class DiagnoseLinearStaticSystem:
    """
    Constructor runs the diagnostics immediately and appends them to
    …/<job_id>/diagnostics/DiagnoseLinearStaticSystem.log
    """

    EIG_LIMIT = 400      # eigen-based extras only for ‘small’ matrices

    # ─────────────────────────────────────────────────────── init ──
    def __init__(
        self,
        *,
        stage: str,
        A_current: Union[np.ndarray, sp.spmatrix],
        b_current: np.ndarray,
        A_full:    Union[np.ndarray, sp.spmatrix],
        b_full:    np.ndarray,
        fixed_dofs: Sequence[int] | np.ndarray,
        condensed_dofs: Sequence[int] | np.ndarray | None,
        job_results_dir: str | Path,
        filename: str = "DiagnoseLinearStaticSystem.log",
    ) -> None:

        self.stage  = stage
        self.A_cur  = A_current
        self.b_cur  = np.asarray(b_current, dtype=float).ravel()
        self.A_full = A_full
        self.b_full = np.asarray(b_full,    dtype=float).ravel()

        self.fixed = np.asarray(fixed_dofs, int) if fixed_dofs is not None else np.empty(0, int)
        self.kept  = np.asarray(condensed_dofs, int) if condensed_dofs is not None else np.empty(0, int)

        # ── directory layout – use diagnostics/ next to maps/  ────────────
        self.job_root  = Path(job_results_dir).resolve()
        self.diag_dir  = self.job_root.parent / "diagnostics"
        self.diag_dir.mkdir(parents=True, exist_ok=True)

        self.log_path  = self.diag_dir / filename

        self._init_logger()
        self._write_block()               # run immediately

    # ─────────────────────────────────────────────────── logging ──
    def _init_logger(self) -> None:
        self.logger = logging.getLogger(f"DiagnoseLinearStaticSystem.{id(self)}")
        if not self.logger.handlers:                         # avoid duplicates
            fh = logging.FileHandler(self.log_path, mode="a", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)

            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            sh.setLevel(logging.INFO)
            self.logger.addHandler(sh)

        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    # ───────────────────────────────────────────────── write block ──
    def _write_block(self) -> None:
        # Which DOFs were completely eliminated (i.e. not kept)?
        if self.kept.size:
            all_dofs   = np.arange(self.A_full.shape[0])
            eliminated = np.setdiff1d(all_dofs, self.kept, assume_unique=True)
        else:
            eliminated = np.empty(0, int)

        cases = {
            "CURRENT":   (self.A_cur,  self.b_cur,  np.empty(0, int), eliminated[:0]),
            "EVOLUTION": (self.A_full, self.b_full, self.fixed,       eliminated),
        }

        stats = {tag: self._collect_stats(*payload) for tag, payload in cases.items()}

        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write("-" * 74 + f"\n### {self.stage}\n\n")
            _twocol(fh, "", "CURRENT", "EVOLUTION")
            for key in stats["CURRENT"]:
                _twocol(fh, key, stats["CURRENT"][key], stats["EVOLUTION"][key])
            fh.write("\n")

        self.logger.info("🩺  %s diagnostics appended → %s", self.stage, self.log_path)

    # ─────────────────────────────────────────── collect statistics ──
    def _collect_stats(
        self,
        A: Union[np.ndarray, sp.spmatrix],
        b: np.ndarray,
        dirichlet: np.ndarray,
        eliminated: np.ndarray,
    ) -> dict[str, str]:
        is_sp = issparse(A)
        n     = A.shape[0]
        nnz   = A.nnz if is_sp else int(np.count_nonzero(A))

        nz_r  = A.getnnz(axis=1) if is_sp else np.count_nonzero(A, axis=1)
        nz_c  = A.getnnz(axis=0) if is_sp else np.count_nonzero(A, axis=0)

        d: dict[str, str] = {}
        d["Equations"]   = str(n)
        d["Dirichlet"]   = str(dirichlet.size)
        d["Eliminated"]  = str(eliminated.size)
        d["Unknowns"]    = str(n - dirichlet.size - eliminated.size)
        d["nnz"]         = str(nnz)
        d["Density"]     = f"{nnz / n**2:.2e}"
        d["Zero rows"]   = str(int((nz_r == 0).sum()))
        d["Zero cols"]   = str(int((nz_c == 0).sum()))

        if nnz:
            rows, cols = (A.nonzero() if is_sp else np.where(A))
            d["Bandwidth"] = str(int(np.abs(rows - cols).max()))
        else:
            d["Bandwidth"] = "0"

        # optional expensive diagnostics
        try:
            dense  = A.toarray() if is_sp else A
            d["Symmetric"] = "yes" if np.allclose(dense, dense.T) else "no"
            s = np.linalg.svd(dense, compute_uv=False)
            rank = int((s > 1e-12 * s[0]).sum())
            d["Rank"]    = str(rank)
            d["Cond-no"] = f"{np.linalg.cond(dense):.1e}"
        except Exception:
            d["Rank"] = d["Cond-no"] = "n/a"
            d.setdefault("Symmetric", "n/a")

        # eigen statistics only for small systems
        if n <= self.EIG_LIMIT and nnz:
            try:
                w = np.linalg.eigvalsh(dense)
                d["λmin/λmax"] = f"{w.min():.1e}/{w.max():.1e}"
                n_pos = int((w >  1e-12).sum())
                n_neg = int((w < -1e-12).sum())
                d["Inertia"]   = f"{n_pos}/{n_neg}/{n - n_pos - n_neg}"
            except Exception:
                d["λmin/λmax"] = d["Inertia"] = "n/a"
        else:
            d["λmin/λmax"] = d["Inertia"] = "—"

        d["b-min/max"] = f"{b.min():.1e}/{b.max():.1e}"
        d["Σb"]        = f"{b.sum():+.1e}"

        if b.size:
            idx = np.argsort(-np.abs(b))[:3]
            d["Top|b|"] = ", ".join(f"{i}:{b[i]:.1e}" for i in idx)
        else:
            d["Top|b|"] = "—"

        return d
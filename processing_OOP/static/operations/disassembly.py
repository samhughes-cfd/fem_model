# processing_OOP/static/operations/disassembly.py

import time, logging
from pathlib import Path
from typing  import Sequence, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix


# ──────────────────────────────── helpers ────────────────────────────────
def _slice_one(dof_map: np.ndarray,
               U: np.ndarray,
               R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vector-slice global vectors → element-local copies."""
    return U[dof_map].copy(), R[dof_map].copy()


def _csr_to_csv(mat: csr_matrix | coo_matrix, path: Path) -> None:
    """Write a sparse matrix as (row, col, value) CSV."""
    if not isinstance(mat, csr_matrix):
        mat = mat.tocsr()
    coo = mat.tocoo()
    pd.DataFrame(
        {"Row": coo.row, "Col": coo.col, "K Value": coo.data}
    ).to_csv(path, index=False)


# ───────────────────────────── main class ────────────────────────────────
class DisassembleGlobalSystem:
    """
    Reverse-assembles global results to each element and writes:

    ├─ elements/
    │   ├─ K_e/00000.csv , …        (one per element – optional)
    │   ├─ F_e/00000.csv , …        (one per element – optional)
    │   ├─ U_e.csv                  (row-wise, all elements)
    │   └─ R_e.csv
    └─ maps/05_disassembly_map.csv
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        *,
        elements: Sequence[object],
        K_mod: csr_matrix,                       # shape-only (sanity)
        F_mod: np.ndarray,
        U_global: np.ndarray,
        R_global: np.ndarray,
        local_global_dof_map: Sequence[np.ndarray],
        element_K_raw: Sequence[csr_matrix | coo_matrix] | None = None,
        element_F_raw: Sequence[np.ndarray] | None = None,
        job_results_dir: str | Path | None = None,
    ) -> None:

        # store ────────────────────────────────────────────────────────────
        self.elements   = list(elements)
        self.K_mod      = K_mod
        self.F_mod      = np.asarray(F_mod, dtype=np.float64).ravel()
        self.U_global   = np.asarray(U_global, dtype=np.float64).ravel()
        self.R_global   = np.asarray(R_global, dtype=np.float64).ravel()
        self.dof_maps   = [np.asarray(m, dtype=np.int32) for m in local_global_dof_map]

        self.element_K_raw = list(element_K_raw) if element_K_raw is not None else None
        self.element_F_raw = list(element_F_raw) if element_F_raw is not None else None

        self.job_results_dir = Path(job_results_dir) if job_results_dir else None

        # logging + quick size check ──────────────────────────────────────
        self.logger = self._init_logging()
        n = self.K_mod.shape[0]
        for name, vec in (("F_mod", self.F_mod),
                          ("U_global", self.U_global),
                          ("R_global", self.R_global)):
            if vec.size != n:
                raise ValueError(f"{name}.size ({vec.size}) ≠ K_mod.shape[0] ({n})")

        for gmap in self.dof_maps:
            if gmap.min() < 0 or gmap.max() >= n:
                raise ValueError("DOF map contains out-of-range indices")

        # outputs ----------------------------------------------------------
        self.U_e: List[np.ndarray] = []
        self.R_e: List[np.ndarray] = []
        self.elapsed: float | None = None

    # -------------------------------------------------------- logging helper
    def _init_logging(self) -> logging.Logger:
        lg = logging.getLogger(f"DisassembleGlobalSystem.{id(self)}")
        lg.handlers.clear()
        lg.setLevel(logging.DEBUG)
        lg.propagate = False

        if self.job_results_dir:
            logs = self.job_results_dir.parent / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(logs / "DisassembleGlobalSystem.log",
                                     mode="w", encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s"))
            lg.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        sh.setLevel(logging.INFO)
        lg.addHandler(sh)
        return lg

    # ------------------------------------------------------------- public API
    def disassemble(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Slice *U_global* and *R_global* for every element; return ``(U_e, R_e)``."""
        t0 = time.perf_counter()
        self.logger.info("🔧 Disassembling global results → element level …")

        self.U_e, self.R_e = map(                               # type: ignore[arg-type]
            list,
            zip(*[_slice_one(m, self.U_global, self.R_global) for m in self.dof_maps])
        )

        self._export_disassembly_map()
        self._export_element_csvs()
        self.elapsed = time.perf_counter() - t0
        self.logger.info("✅ Disassembly finished in %.2fs", self.elapsed)
        return self.U_e, self.R_e

    # -------------------------------------------------------------- CSV/write
    def _export_disassembly_map(self) -> None:
        if self.job_results_dir is None:
            return
        maps = self.job_results_dir.parent / "maps"
        maps.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {
                "Element ID": range(len(self.dof_maps)),
                "Global DOF": [str(m.tolist()) for m in self.dof_maps],
                "Local DOF" : [str(list(range(m.size))) for m in self.dof_maps],
            }
        ).to_csv(maps / "05_disassembly_map.csv", index=False)
        self.logger.info("🗺️  Disassembly map saved")

    def _export_element_csvs(self) -> None:
        if self.job_results_dir is None:
            return

        base = self.job_results_dir / "elements"
        (base / "K_e").mkdir(parents=True, exist_ok=True)
        (base / "F_e").mkdir(parents=True, exist_ok=True)

        # ── per-element Kₑ & Fₑ ------------------------------------------
        for eid in range(len(self.elements)):
            tag = f"{eid:05d}.csv"
            if self.element_K_raw:
                _csr_to_csv(self.element_K_raw[eid], base / "K_e" / tag)
            if self.element_F_raw:
                pd.DataFrame(
                    {
                        "Local DOF": range(len(self.element_F_raw[eid])),
                        "F Value"  : self.element_F_raw[eid],
                    }
                ).to_csv(base / "F_e" / tag, index=False)

        # ── merged Uₑ / Rₑ -----------------------------------------------
        def _stack(list_of_arrs: List[np.ndarray], label: str) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "Element ID": np.repeat(
                        range(len(list_of_arrs)),
                        [len(a) for a in list_of_arrs]),
                    "Local DOF" : np.concatenate([np.arange(len(a)) for a in list_of_arrs]),
                    label       : np.concatenate(list_of_arrs),
                }
            )

        _stack(self.U_e, "U Value").to_csv(base / "U_e.csv", index=False)
        _stack(self.R_e, "R Value").to_csv(base / "R_e.csv", index=False)
        self.logger.info("💾 Element CSVs written → %s", base)
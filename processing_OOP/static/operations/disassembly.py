# processing_OOP/static/operations/disassembly.py

import time, logging
from pathlib import Path
from typing import Sequence, List, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────── helpers ────────────────────────────────
def _slice_one(dof_map: np.ndarray,
               U: np.ndarray,
               R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vector-slice global vectors → element-local copies."""
    return U[dof_map].copy(), R[dof_map].copy()


# ───────────────────────────── main class ────────────────────────────────
class DisassembleGlobalSystem:
    """
    Reverse-assembles global results to each element and writes:

    ├─ elements/
    │   ├─ U_e/00000.csv , …
    │   ├─ R_e/00000.csv , …
    │   └─ R_residual_e/00000.csv (if F_e is passed)
    └─ maps/05_disassembly_map.csv
    """

    def __init__(
        self,
        *,
        U_global: np.ndarray,
        R_global: np.ndarray,
        R_residual: np.ndarray,
        local_global_dof_map: Sequence[np.ndarray],
        job_results_dir: str | Path | None = None,
        F_e: Sequence[np.ndarray] | None = None, 
    ) -> None:

        self.U_global = np.asarray(U_global, dtype=np.float64).ravel()
        self.R_global = np.asarray(R_global, dtype=np.float64).ravel()
        self.R_residual = np.asarray(R_residual, dtype=np.float64).ravel()
        self.dof_maps = [np.asarray(m, dtype=np.int32) for m in local_global_dof_map]

        self.job_results_dir = Path(job_results_dir) if job_results_dir else None
        self.F_e = F_e 
        self.logger = self._init_logging()

        n = self.U_global.size

        for name, vec in (("U_global", self.U_global),
                          ("R_global", self.R_global)):
            if vec.size != n:
                raise ValueError(f"{name}.size ({vec.size}) ≠ K_mod.shape[0] ({n})")

        for gmap in self.dof_maps:
            if gmap.min() < 0 or gmap.max() >= n:
                raise ValueError("DOF map contains out-of-range indices")

        self.U_e: List[np.ndarray] = []
        self.R_e: List[np.ndarray] = []
        self.R_residual_e: List[np.ndarray] = []
        self.elapsed: float | None = None

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

    def disassemble(self, F_e: Sequence[np.ndarray] | None = None) -> Tuple[
        List[np.ndarray],  # U_e
        List[np.ndarray],  # R_e
        List[np.ndarray],  # R_residual_e
    ]:
        """Disassemble global vectors to element-local U, R, and optionally compute residuals."""
        t0 = time.perf_counter()
        self.logger.info("🔧 Disassembling global results → element level …")

        self.U_e, self.R_e = map(
            list,
            zip(*[_slice_one(m, self.U_global, self.R_global) for m in self.dof_maps])
        )

        # 🔧 Use passed F_e if given, else fallback to constructor-stored
        F_e = F_e if F_e is not None else self.F_e

        if F_e:
            self.R_residual_e = [
                R_e - F_e for R_e, F_e in zip(self.R_e, F_e)
            ]
        else:
            self.logger.warning("⚠️ F_e not provided — residuals will not be computed.")

        self._export_disassembly_map()
        self._export_element_csvs()
        self.elapsed = time.perf_counter() - t0
        self.logger.info("✅ Disassembly finished in %.2fs", self.elapsed)

        return self.U_e, self.R_e, self.R_residual_e

    def _export_disassembly_map(self) -> None:
        if self.job_results_dir is None:
            return

        maps = self.job_results_dir.parent / "maps"
        maps.mkdir(parents=True, exist_ok=True)

        #pd.DataFrame(
            #{
                #"Element ID": range(len(self.dof_maps)),
                #"Global DOF": [str(m.tolist()) for m in self.dof_maps],
                #"Local DOF": [str(list(range(m.size))) for m in self.dof_maps],
            #}
        #).to_csv(maps / "05_disassembly_map.csv", index=False)
        #self.logger.info("🗺️  Disassembly map saved")

    def _export_element_csvs(self) -> None:
        if self.job_results_dir is None:
            return

        #base = self.job_results_dir / "elements"
        #(base / "U_e").mkdir(parents=True, exist_ok=True)
        #(base / "R_e").mkdir(parents=True, exist_ok=True)
        #(base / "R_residual_e").mkdir(parents=True, exist_ok=True)

        #for eid in range(len(self.dof_maps)):
            #tag = f"{eid:05d}.csv"

            #pd.DataFrame({
                #"Local DOF": np.arange(len(self.U_e[eid])),
                #"U Value": self.U_e[eid],
            #}).to_csv(base / "U_e" / tag, index=False)

            #pd.DataFrame({
                #"Local DOF": np.arange(len(self.R_e[eid])),
                #"R Value": self.R_e[eid],
           #}).to_csv(base / "R_e" / tag, index=False)

            #if self.R_residual_e:
                #pd.DataFrame({
                    #"Local DOF": np.arange(len(self.R_residual_e[eid])),
                    #"Residual Value": self.R_residual_e[eid],
                #}).to_csv(base / "R_residual_e" / tag, index=False)

        #self.logger.info("💾 Element CSVs written → %s", base)
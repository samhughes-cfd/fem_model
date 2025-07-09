"""Load‑case visualisation utility (July 2025).

• **Updated** to use the new ``GridParser`` instead of the legacy mesh parser.
• Mirrors ``VisualiseDeformation`` structure and path discovery.
• Generates six‑component force/moment diagrams for every load file.
• X‑axis scaling: two black reference markers (x = 0, x = L) force the
  natural padding without hard‑clipping or manual margin fiddling.
"""

from __future__ import annotations

import datetime as _dt
import re
import sys
from pathlib import Path
from typing import Callable, Final, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------#
#  Project paths
# ---------------------------------------------------------------------------#
SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = next(
    (p for p in SCRIPT_DIR.parents if (p / "pre_processing").is_dir()),
    SCRIPT_DIR.parents[4],
)
sys.path.append(str(PROJECT_ROOT))  # local imports

# --- External parsers ------------------------------------------------------#
from pre_processing.parsing.grid_parser import GridParser  # type: ignore
from pre_processing.parsing.distributed_load_parser import (  # type: ignore
    parse_distributed_load,
)
from pre_processing.parsing.point_load_parser import parse_point_load  # type: ignore


class VisualiseLoad:
    """Produce force & moment diagrams for every load file in each job."""

    _BLUE: Final[str] = "#4F81BD"

    def __init__(self) -> None:
        self.figure_output_dir: Final[Path] = SCRIPT_DIR / "load_plots"
        self.figure_output_dir.mkdir(exist_ok=True)

        self.jobs_dir: Final[Path] = PROJECT_ROOT / "jobs"
        self._parsers: Final[Mapping[str, Callable[[Path], np.ndarray]]] = {
            "point": parse_point_load,
            "distributed": parse_distributed_load,
        }

    # ------------------------------------------------------------------#
    #  Grid‑helper (shared with deformation visualiser)
    # ------------------------------------------------------------------#
    @staticmethod
    def _get_node_coordinates(grid_obj: object) -> np.ndarray:
        """Return the (N, 3) array of node coordinates from a GridParser result."""
        # 1️⃣ Official nested layout
        if isinstance(grid_obj, dict) and "grid_dictionary" in grid_obj:
            inner = grid_obj["grid_dictionary"]
            if isinstance(inner, dict) and "coordinates" in inner:
                return inner["coordinates"]  # type: ignore[index]

        # 2️⃣ Optional flat / attribute fall‑backs
        if isinstance(grid_obj, dict) and "node_coordinates" in grid_obj:
            return grid_obj["node_coordinates"]  # type: ignore[index]
        if hasattr(grid_obj, "node_coordinates"):
            return getattr(grid_obj, "node_coordinates")  # type: ignore[arg-type]

        raise KeyError("grid data does not contain 'grid_dictionary' → 'coordinates'")

    # ------------------------------------------------------------------#
    #  Plot helper
    # ------------------------------------------------------------------#
    def _plot(
        self,
        load: np.ndarray,
        *,
        load_type: str,
        L: float,
        title_suffix: str,
        save_path: Path,
    ) -> None:
        """Plot 3 force + 3 moment profiles, anchoring x at 0 and L."""
        if load.shape[1] != 9:
            raise ValueError("load array must be (n, 9) [x,y,z,Fx,Fy,Fz,Mx,My,Mz]")

        x = load[:, 0]
        F = load[:, 3:6]
        M = load[:, 6:9]

        labels = [
            r"$F_x\,[\mathrm{N}]$",
            r"$F_y\,[\mathrm{N}]$",
            r"$F_z\,[\mathrm{N}]$",
            r"$M_x\,[\mathrm{N\,m}]$",
            r"$M_y\,[\mathrm{N\,m}]$",
            r"$M_z\,[\mathrm{N\,m}]$",
        ]

        fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
        fig.suptitle(
            f"{load_type.capitalize()} load – {title_suffix}",
            fontsize=16,
            fontweight="bold",
        )

        # iterate rows: Fx/Fy/Fz and Mx/My/Mz
        for i, (ax_F, ax_M) in enumerate(zip(axes[:, 0], axes[:, 1])):
            # --- Forces ---
            if load_type == "distributed":
                ax_F.plot(x, F[:, i], color=self._BLUE, linewidth=2)
                ax_F.fill_between(x, F[:, i], 0, color=self._BLUE, alpha=0.25)
            else:  # point loads
                for xi, yi in zip(x, F[:, i]):
                    if yi == 0:
                        continue
                    ax_F.plot([xi, xi], [0, yi], color=self._BLUE, linewidth=2)
                    ax_F.plot(xi, yi, marker="v", color=self._BLUE, markersize=8)

            # --- Moments ---
            if load_type == "distributed":
                ax_M.plot(x, M[:, i], color=self._BLUE, linewidth=2)
                ax_M.fill_between(x, M[:, i], 0, color=self._BLUE, alpha=0.25)
            else:
                for xi, yi in zip(x, M[:, i]):
                    if yi == 0:
                        continue
                    ax_M.plot([xi, xi], [0, yi], color=self._BLUE, linewidth=2)
                    ax_M.plot(xi, yi, marker="v", color=self._BLUE, markersize=8)

            # --- anchor beam ends (x = 0 & x = L) to drive axis scaling ---
            for ax in (ax_F, ax_M):
                ax.plot([0], [0], marker="o", color="k", markersize=3)
                ax.plot([L], [0], marker="o", color="k", markersize=3)
                ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
                ax.grid(ls="--", alpha=0.6)

            # labels & titles
            ax_F.set_ylabel(labels[i])
            ax_M.set_ylabel(labels[i + 3])
            if i == 0:
                ax_F.set_title("Forces", fontweight="bold")
                ax_M.set_title("Moments", fontweight="bold")

        axes[-1, 0].set_xlabel(r"$x\,[\mathrm{m}]$")
        axes[-1, 1].set_xlabel(r"$x\,[\mathrm{m}]$")

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    # ------------------------------------------------------------------#
    #  Driver
    # ------------------------------------------------------------------#
    def process_all(self) -> None:
        """Scan each job_* folder and visualise its load files."""
        for job_dir in sorted(self.jobs_dir.glob("job_*")):
            m = re.match(r"job_(\d+)", job_dir.name)
            if not m:
                continue
            job_id = m.group(1)

            # --- Beam length L from grid (preferred) or load x‑span (fallback) ---
            L: Optional[float] = None
            grid_file = job_dir / "grid.txt"
            if grid_file.is_file():
                try:
                    grid = GridParser(str(grid_file), str(job_dir)).parse()
                    node_coords = self._get_node_coordinates(grid)
                    xs = node_coords[:, 0]
                    L = float(xs.max() - xs.min())
                except Exception:
                    pass  # fall back later

            for load_type, parser in self._parsers.items():
                load_file = job_dir / f"{load_type}_load.txt"
                if not load_file.is_file():
                    continue

                try:
                    load_arr = parser(load_file)
                except Exception as exc:
                    print(f"⚠️  job_{job_id}: cannot parse {load_type} load – {exc}")
                    continue

                if L is None or L <= 0:
                    xs = load_arr[:, 0]
                    L = float(xs.max() - xs.min()) if xs.size else 1.0

                ts = _dt.datetime.fromtimestamp(
                    load_file.stat().st_mtime
                ).strftime("%Y-%m-%d_%H-%M-%S")
                fig_name = f"load_job_{job_id}_{ts}.png"

                self._plot(
                    load_arr,
                    load_type=load_type,
                    L=L,
                    title_suffix=f"job_{job_id}_{ts}",
                    save_path=self.figure_output_dir / fig_name,
                )


if __name__ == "__main__":
    VisualiseLoad().process_all()

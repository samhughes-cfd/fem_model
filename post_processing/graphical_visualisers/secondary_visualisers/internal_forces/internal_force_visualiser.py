# post_processing/graphical_visualisers/secondary_visualisers/internal_forces/internal_force_visualisation.py
"""Internal-force visualisation utility (N, Vₙ, Mₙ).

* Mirrors the structure of VisualiseDeformation / VisualiseLoad.
* Reads each *_IF_global.csv produced in primary_results/.
* Outputs PNGs to internal_force_plots/.
"""

from __future__ import annotations

import glob
import re
import sys
import datetime as _dt
from pathlib import Path
from typing import Final, Optional

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------#
#  Repository paths
# ---------------------------------------------------------------------------#
SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = next(
    (p for p in SCRIPT_DIR.parents if (p / "pre_processing").is_dir()),
    SCRIPT_DIR.parents[4],
)
sys.path.append(str(PROJECT_ROOT))

# Local parser
from pre_processing.parsing.mesh_parser import parse_mesh  # type: ignore


class VisualiseInternalForces:
    """Create six internal-force diagrams for every job."""

    _BLUE: Final[str] = "#4F81BD"

    def __init__(self) -> None:
        # Where result CSVs live
        self.results_dir: Final[Path] = PROJECT_ROOT / "post_processing" / "results"
        self.jobs_dir:    Final[Path] = PROJECT_ROOT / "jobs"

        # Figure output directory
        self.fig_dir: Final[Path] = SCRIPT_DIR / "internal_force_plots"
        self.fig_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------#
    #  Plot helper
    # ------------------------------------------------------------------#
    def _plot(
        self,
        forces: np.ndarray,
        x: np.ndarray,
        L: float,
        *,
        title_suffix: str,
        save_path: Path,
    ) -> None:
        """Draw N, Vy, Vz, Mx, My, Mz on a 3×2 grid with beam-end anchors."""
        if forces.shape[1] != 6:
            raise ValueError("expect (n,6) → N Vy Vz Mx My Mz")

        N, Vy, Vz, Mx, My, Mz = forces.T
        labels = [
            r"$N\,[\mathrm{N}]$",
            r"$V_y\,[\mathrm{N}]$",
            r"$V_z\,[\mathrm{N}]$",
            r"$M_x\,[\mathrm{N\,m}]$",
            r"$M_y\,[\mathrm{N\,m}]$",
            r"$M_z\,[\mathrm{N\,m}]$",
        ]
        series = [(N, labels[0]), (Mx, labels[3]),
                  (Vy, labels[1]), (My, labels[4]),
                  (Vz, labels[2]), (Mz, labels[5])]

        fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
        fig.suptitle(f"Internal-force diagrams – {title_suffix}",
                     fontsize=16, fontweight="bold")

        for ax, (y, lbl) in zip(axes.ravel(order="F"), series):
            ax.plot(x, y, color=self._BLUE, linewidth=2)
            # Shear & bending look nice filled:
            if "V_" in lbl or "M_" in lbl:
                ax.fill_between(x, y, 0, color=self._BLUE, alpha=0.25)

            # Anchor beam ends to drive axis autoscale + whitespace
            ax.plot([0], [0], "ko", ms=3)
            ax.plot([L], [0], "ko", ms=3)

            ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
            ax.grid(ls="--", alpha=0.6)
            ax.set_ylabel(lbl)

        axes[-1, 0].set_xlabel(r"$x\,[\mathrm{m}]$")
        axes[-1, 1].set_xlabel(r"$x\,[\mathrm{m}]$")

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    # ------------------------------------------------------------------#
    #  Main driver
    # ------------------------------------------------------------------#
    def process_all(self) -> None:
        pattern = self.results_dir / "job_*" / "primary_results" / "*_IF_global.csv"
        csv_files = sorted(glob.glob(str(pattern)))
        if not csv_files:
            print("No *_IF_global.csv files found.")
            return

        for csv_path in csv_files:
            csv = Path(csv_path)
            job_dir = csv.parent.parent                          # …/job_xxx/…
            m = re.match(r"job_(\d+)_(\d{4}-\d{2}-\d{2}_[\d\-]+)", job_dir.name)
            if not m:
                print(f"Skipping folder {job_dir.name}")
                continue
            job_id, timestamp = m.groups()

            # Mesh → node x positions → beam length
            mesh_file = self.jobs_dir / f"job_{job_id}" / "mesh.txt"
            try:
                mesh = parse_mesh(mesh_file)
                node_x = mesh["node_coordinates"][:, 0]
                L = float(node_x.max() - node_x.min())
            except Exception as exc:
                print(f"⚠️  job_{job_id}: mesh problem – {exc}")
                continue

            # Read internal-force CSV (skip first col = x from solver)
            try:
                full = np.genfromtxt(csv, delimiter=",", skip_header=1)
                if full.shape[1] < 7:
                    raise ValueError("expected ≥7 columns")
                forces = full[:, 1:7]
            except Exception as exc:
                print(f"⚠️  {csv.name}: {exc}")
                continue

            fig_name = f"internal_job_{job_id}_{timestamp}.png"
            self._plot(
                forces,
                node_x,
                L,
                title_suffix=f"job_{job_id}_{timestamp}",
                save_path=self.fig_dir / fig_name,
            )


if __name__ == "__main__":
    VisualiseInternalForces().process_all()
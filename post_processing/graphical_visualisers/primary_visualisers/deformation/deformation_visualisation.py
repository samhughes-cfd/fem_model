# post_processing/graphical_visualisers/deformation/deformation_visualisation.py

"""Deformation visualisation utility.

Updated June 2025 for new results directory layout, *_U_global.csv format,
**and** corrected parent‑folder detection (job folder now two levels above
primary_results).
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path
from typing import Final, Optional

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------#
#  Project paths
# ---------------------------------------------------------------------------#
SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent

# Walk upwards until we find the repo root (must contain pre_processing)
PROJECT_ROOT: Final[Path] = next(
    (p for p in SCRIPT_DIR.parents if (p / "pre_processing").is_dir()),
    SCRIPT_DIR.parents[4],
)

sys.path.append(str(PROJECT_ROOT))

# Local import after sys.path tweak
from pre_processing.parsing.mesh_parser import parse_mesh  # type: ignore

# ---------------------------------------------------------------------------#
#  Visualiser class
# ---------------------------------------------------------------------------#

class VisualiseDeformation:
    """Produce translation / rotation profiles from *_U_global.csv files."""

    def __init__(self) -> None:
        self.figure_output_dir: Final[Path] = SCRIPT_DIR / "deformation_plots"
        self.figure_output_dir.mkdir(exist_ok=True)

        self.base_dir: Final[Path] = PROJECT_ROOT / "post_processing" / "results"
        self.mesh_dir: Final[Path] = PROJECT_ROOT / "jobs"

    # ---------------------------------------------------------------------#
    #  Plotting
    # ---------------------------------------------------------------------#

    def _plot(
        self,
        U: np.ndarray,
        node_positions: np.ndarray,
        *,
        scale: float = 1.0,
        title_suffix: str = "",
        save_path: Optional[Path] = None,
    ) -> None:
        if U.shape[1] != 6:
            raise ValueError("U must be (n_nodes, 6)")

        fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
        fig.suptitle(
            rf"Raw $U_g(x)${' – ' + title_suffix if title_suffix else ''}",
            fontsize=16,
            fontweight="bold",
        )

        color = "#4F81BD"
        pairs = [
            (U[:, 0] * 1_000 * scale, r"$u_x(x)\ \mathrm{[mm]}$", U[:, 3], r"$\theta_x(x)\ [^\circ]$"),
            (U[:, 1] * 1_000 * scale, r"$u_y(x)\ \mathrm{[mm]}$", U[:, 5], r"$\theta_z(x)\ [^\circ]$"),
            (U[:, 2] * 1_000 * scale, r"$u_z(x)\ \mathrm{[mm]}$", U[:, 4], r"$\theta_y(x)\ [^\circ]$"),
        ]


        for i, (ax_l, ax_r, (disp, disp_lbl, rot, rot_lbl)) in enumerate(
            zip(axes[:, 0], axes[:, 1], pairs)
        ):
            ax_l.plot(node_positions, disp, color=color, marker="o")
            ax_l.axhline(0, color="k", linestyle="--", linewidth=0.8)
            ax_l.grid(ls="--", alpha=0.6)
            ax_l.set_ylabel(disp_lbl)

            ax_r.plot(node_positions, np.degrees(rot) * scale, color=color, marker="o")
            ax_r.axhline(0, color="k", linestyle="--", linewidth=0.8)
            ax_r.grid(ls="--", alpha=0.6)
            ax_r.set_ylabel(rot_lbl)

            if i == 0:
                ax_l.set_title("Translation profiles", fontweight="bold")
                ax_r.set_title("Rotation profiles", fontweight="bold")

        axes[-1, 0].set_xlabel(r"$x$ [m]")
        axes[-1, 1].set_xlabel(r"$x$ [m]")
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        if save_path:
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()

    # ---------------------------------------------------------------------#
    #  File helpers
    # ---------------------------------------------------------------------#

    @staticmethod
    def _read_U_global(file: Path) -> Optional[np.ndarray]:
        try:
            vals = np.genfromtxt(file, delimiter=",", skip_header=1, usecols=1)
            if vals.size % 6:
                raise ValueError("DOF count not divisible by 6")
            return vals.reshape(-1, 6)
        except Exception as exc:
            print(f"Error reading {file}: {exc}")
            return None

    # ---------------------------------------------------------------------#
    #  Main driver
    # ---------------------------------------------------------------------#

    def process_all(self) -> None:
        pattern = self.base_dir / "job_*" / "primary_results" / "*_U_global.csv"
        files = sorted(glob.glob(str(pattern)))
        if not files:
            print("No deformation files found.")
            return

        for file_path in files:
            file = Path(file_path)
            job_dir = file.parent.parent  # …/job_xxx/.../primary_results/ <─ two levels up
            job_name = job_dir.name

            m = re.match(
                r"job_(?P<id>\d+)_(?P<ts>\d{4}-\d{2}-\d{2}_[\d\-]+)",
                job_name,
            )
            if not m:
                print(f"Skipping unrecognised folder '{job_name}'")
                continue

            job_id = m.group("id")
            timestamp = m.group("ts")

            mesh_file = self.mesh_dir / f"job_{job_id}" / "mesh.txt"
            print(f"→ Processing job {job_id} ({timestamp})")

            U = self._read_U_global(file)
            mesh = parse_mesh(mesh_file) if mesh_file.is_file() else None

            if U is None or mesh is None or "node_coordinates" not in mesh:
                print(f"⚠️  Missing data for job {job_id}, skipping.")
                continue

            node_x = mesh["node_coordinates"][:, 0]
            fig_name = f"deformation_job_{job_id}_{timestamp}.png"
            self._plot(
                U,
                node_x,
                title_suffix=f"job_{job_id}_{timestamp}",
                save_path=self.figure_output_dir / fig_name,
            )


if __name__ == "__main__":
    VisualiseDeformation().process_all()
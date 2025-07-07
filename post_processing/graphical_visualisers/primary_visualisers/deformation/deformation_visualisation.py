"""
Deformation visualisation utility – July 2025
---------------------------------------------

Reads every *_U_global.csv under post_processing/results/**/primary_results
and produces translation / rotation profiles.  Mesh geometry is supplied by
``grid.txt`` files parsed with ``GridParser``; each parser result is expected
to look like::

    {
        "grid_dictionary": {
            "ids":         ndarray[int64],
            "coordinates": ndarray[float64]  # shape (N, 3)
        }
    }
"""

from __future__ import annotations

import glob
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

# --- grid parser -----------------------------------------------------------#
from pre_processing.parsing.grid_parser import GridParser  # type: ignore


# ---------------------------------------------------------------------------#
#  Visualiser
# ---------------------------------------------------------------------------#
class VisualiseDeformation:
    """Produce translation / rotation profiles from *_U_global.csv files."""

    _BLUE: Final[str] = "#4F81BD"

    def __init__(self) -> None:
        self.figure_output_dir: Final[Path] = SCRIPT_DIR / "deformation_plots"
        self.figure_output_dir.mkdir(exist_ok=True)

        self.results_dir: Final[Path] = PROJECT_ROOT / "post_processing" / "results"
        self.jobs_dir: Final[Path] = PROJECT_ROOT / "jobs"

    # ------------------------------------------------------------------#
    #  Internal helpers
    # ------------------------------------------------------------------#
    @staticmethod
    def _get_node_coordinates(grid_obj: object) -> np.ndarray:
        """
        Extract the (N, 3) array of node coordinates from the object returned
        by ``GridParser.parse()``.

        Expected shape (dict only):

            grid_obj["grid_dictionary"]["coordinates"]

        Falls back to ``.node_coordinates`` or ``["node_coordinates"]`` if they
        ever appear in a future refactor.
        """
        # 1️⃣ official / nested layout
        if isinstance(grid_obj, dict) and "grid_dictionary" in grid_obj:
            inner = grid_obj["grid_dictionary"]
            if isinstance(inner, dict) and "coordinates" in inner:
                return inner["coordinates"]  # type: ignore[index]

        # 2️⃣ optional flat / attribute fall-backs
        if isinstance(grid_obj, dict) and "node_coordinates" in grid_obj:
            return grid_obj["node_coordinates"]  # type: ignore[index]
        if hasattr(grid_obj, "node_coordinates"):
            return getattr(grid_obj, "node_coordinates")  # type: ignore[arg-type]

        raise KeyError(
            "grid data does not contain 'grid_dictionary' → 'coordinates'"
        )

    # ------------------------------------------------------------------#
    #  Plotting
    # ------------------------------------------------------------------#
    def _plot(
        self,
        U: np.ndarray,
        node_positions: np.ndarray,
        *,
        scale: float = 1.0,
        title_suffix: str = "",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot raw global displacement/rotation with beam-end anchors."""
        if U.shape[1] != 6:
            raise ValueError("U must be shaped (n_nodes, 6)")

        x_min, x_max = float(node_positions.min()), float(node_positions.max())

        fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
        fig.suptitle(
            rf"Raw $U_g(x)${' – ' + title_suffix if title_suffix else ''}",
            fontsize=16,
            fontweight="bold",
        )

        pairs = [
            (
                U[:, 0] * 1_000 * scale,
                r"$u_x(x)\ \mathrm{[mm]}$",
                U[:, 3],
                r"$\theta_x(x)\ [^\circ]$",
            ),
            (
                U[:, 1] * 1_000 * scale,
                r"$u_y(x)\ \mathrm{[mm]}$",
                U[:, 5],
                r"$\theta_z(x)\ [^\circ]$",
            ),
            (
                U[:, 2] * 1_000 * scale,
                r"$u_z(x)\ \mathrm{[mm]}$",
                U[:, 4],
                r"$\theta_y(x)\ [^\circ]$",
            ),
        ]

        for i, (ax_l, ax_r, (disp, disp_lbl, rot, rot_lbl)) in enumerate(
            zip(axes[:, 0], axes[:, 1], pairs)
        ):
            # translations
            ax_l.plot(node_positions, disp, color=self._BLUE, marker="o")
            # rotations
            ax_r.plot(node_positions, np.degrees(rot) * scale, color=self._BLUE, marker="o")

            # beam-end anchors + baseline
            for ax in (ax_l, ax_r):
                ax.plot([x_min], [0], marker="o", color="k", ms=3, zorder=3)
                ax.plot([x_max], [0], marker="o", color="k", ms=3, zorder=3)
                ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
                ax.grid(ls="--", alpha=0.6)

            ax_l.set_ylabel(disp_lbl)
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

    # ------------------------------------------------------------------#
    #  CSV helper
    # ------------------------------------------------------------------#
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

    # ------------------------------------------------------------------#
    #  Driver
    # ------------------------------------------------------------------#
    def process_all(self) -> None:
        pattern = self.results_dir / "job_*" / "primary_results" / "*_U_global.csv"
        csv_files = sorted(glob.glob(str(pattern)))
        if not csv_files:
            print("No deformation files found.")
            return

        for csv_path in csv_files:
            csv_file = Path(csv_path)
            # …/results/job_123_2025-06-30_14-52-01/primary_results/xyz_U_global.csv
            job_dir = csv_file.parent.parent
            m = re.match(r"job_(?P<id>\d+)_(?P<ts>[\d\-_]+)", job_dir.name)
            if not m:
                print(f"Skipping unrecognised folder '{job_dir.name}'")
                continue

            job_id, timestamp = m.group("id"), m.group("ts")
            grid_file = self.jobs_dir / f"job_{job_id}" / "grid.txt"
            print(f"→ Processing job {job_id} ({timestamp})")

            # ---- Displacements ------------------------------------------ #
            U = self._read_U_global(csv_file)
            if U is None:
                print(f"⚠️  Could not read displacements for job {job_id}, skipping.")
                continue

            # ---- Geometry (grid) ---------------------------------------- #
            if not grid_file.is_file():
                print(f"⚠️  Grid file missing for job {job_id}, skipping.")
                continue

            grid = GridParser(str(grid_file), str(job_dir)).parse()
            try:
                node_coords = self._get_node_coordinates(grid)
            except (AttributeError, KeyError) as exc:
                print(f"⚠️  {exc} for job {job_id}, skipping.")
                continue

            # ---- Plot ---------------------------------------------------- #
            fig_name = f"deformation_job_{job_id}_{timestamp}.png"
            self._plot(
                U,
                node_coords[:, 0],
                title_suffix=f"job_{job_id}_{timestamp}",
                save_path=self.figure_output_dir / fig_name,
            )


if __name__ == "__main__":
    VisualiseDeformation().process_all()
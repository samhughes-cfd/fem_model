# post_processing/validation_visualisers/verification/deformation_convergence.py

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

PROJECT_ROOT: Final[Path] = next(
    (p for p in SCRIPT_DIR.parents if (p / "pre_processing").is_dir()),
    SCRIPT_DIR.parents[4],
)
sys.path.append(str(PROJECT_ROOT))

from pre_processing.parsing.grid_parser import GridParser  # type: ignore


class VisualiseDeformationConvergence:
    """
    Overlay translation / rotation profiles from *_U_global.csv files
    on a single figure, assuming each belongs to a convergence study.
    """

    _LINESTYLES: Final[list[str]] = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    _COLORS: Final[list[str]] = ["#4F81BD", "#C0504D", "#9BBB59", "#8064A2", "#4BACC6"]

    def __init__(self) -> None:
        self.figure_output_dir: Final[Path] = SCRIPT_DIR / "deformation_plots"
        self.figure_output_dir.mkdir(exist_ok=True)

        self.results_dir: Final[Path] = PROJECT_ROOT / "post_processing" / "results"
        self.jobs_dir: Final[Path] = PROJECT_ROOT / "jobs"

    @staticmethod
    def _get_node_coordinates(grid_obj: object) -> np.ndarray:
        if isinstance(grid_obj, dict) and "grid_dictionary" in grid_obj:
            inner = grid_obj["grid_dictionary"]
            if isinstance(inner, dict) and "coordinates" in inner:
                return inner["coordinates"]
        if isinstance(grid_obj, dict) and "node_coordinates" in grid_obj:
            return grid_obj["node_coordinates"]
        if hasattr(grid_obj, "node_coordinates"):
            return getattr(grid_obj, "node_coordinates")
        raise KeyError("grid data does not contain 'grid_dictionary' → 'coordinates'")

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

    def process_convergence_plot(self, scale: float = 1.0) -> None:
        pattern = str(self.results_dir / "job_*" / "primary_results" / "global" / "U_global.csv")
        csv_files = sorted(glob.glob(pattern))
        if not csv_files:
            print("No deformation files found.")
            return

        fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
        fig.suptitle("Convergence Plot: $U_g(x)$ across jobs", fontsize=16, fontweight="bold")

        pairs = [
            (0, r"$u_x(x)\ \mathrm{[mm]}$", 3, r"$\theta_x(x)\ [^\circ]$"),
            (1, r"$u_y(x)\ \mathrm{[mm]}$", 5, r"$\theta_z(x)\ [^\circ]$"),
            (2, r"$u_z(x)\ \mathrm{[mm]}$", 4, r"$\theta_y(x)\ [^\circ]$"),
        ]

        x_ref = None
        all_data = []

        for idx, csv_path in enumerate(csv_files):
            csv_file = Path(csv_path)
            job_dir = csv_file.parent.parent.parent
            m = re.match(r"job_(?P<id>\d+)_(?P<ts>[\d\-_]+_pid\d+_[a-f0-9]+)", job_dir.name)
            if not m:
                print(f"Skipping unrecognised folder '{job_dir.name}'")
                continue

            job_id, timestamp = m.group("id"), m.group("ts")
            label = f"job_{job_id}"
            line_style = self._LINESTYLES[idx % len(self._LINESTYLES)]
            color = self._COLORS[idx % len(self._COLORS)]

            grid_file = self.jobs_dir / f"job_{job_id}" / "grid.txt"
            if not grid_file.is_file():
                print(f"⚠️ Missing grid file for job {job_id}, skipping.")
                continue

            U = self._read_U_global(csv_file)
            if U is None:
                print(f"⚠️ Failed to read displacements for job {job_id}, skipping.")
                continue

            grid = GridParser(str(grid_file), str(job_dir)).parse()
            try:
                node_coords = self._get_node_coordinates(grid)
            except Exception as exc:
                print(f"⚠️ {exc} for job {job_id}, skipping.")
                continue

            x = node_coords[:, 0]
            if x.shape[0] != U.shape[0]:
                print(f"❌ Mismatch: x.shape = {x.shape}, U.shape = {U.shape} for job {job_id}")
                continue

            if x_ref is None or len(x) > len(x_ref):
                x_ref = x  # finest mesh for analytical

            ux = U[:, 0] * 1000 * scale
            uy = U[:, 1] * 1000 * scale
            uz = U[:, 2] * 1000 * scale
            thetax = np.degrees(U[:, 3]) * scale
            thetay = np.degrees(U[:, 4]) * scale
            thetaz = np.degrees(U[:, 5]) * scale

            for i, (disp_idx, _, rot_idx, _) in enumerate(pairs):
                axes[i, 0].plot(x, U[:, disp_idx] * 1000 * scale, label=label, color=color, linestyle=line_style)
                axes[i, 1].plot(x, np.degrees(U[:, rot_idx]) * scale, label=label, color=color, linestyle=line_style)

            job_col = np.full_like(x, int(job_id), dtype=int).reshape(-1, 1)
            block = np.column_stack([job_col, x, ux, uy, uz, thetax, thetay, thetaz])
            all_data.append(block)

        uy_analytical = None
        thetaz_analytical = None

        if x_ref is not None:
            F = -500        # Load [N]
            E = 2e11        # Young's modulus [Pa]
            I_z = 2.08769e-06  # Moment of inertia [m^4]
            L = float(np.max(x_ref))

            uy_analytical = (F * x_ref**2) / (6 * E * I_z) * (3 * L - x_ref)
            thetaz_analytical = (F * x_ref) / (6 * E * I_z) * (6 * L - 3 * x_ref)

            axes[1, 0].plot(x_ref, uy_analytical * 1000, 'k--', label="Analytical")
            axes[1, 1].plot(x_ref, np.degrees(thetaz_analytical), 'k--', label="Analytical")

        for i, (_, disp_lbl, _, rot_lbl) in enumerate(pairs):
            axes[i, 0].set_ylabel(disp_lbl)
            axes[i, 1].set_ylabel(rot_lbl)
            axes[i, 0].grid(ls="--", alpha=0.6)
            axes[i, 1].grid(ls="--", alpha=0.6)

            if i == 0:
                axes[i, 0].set_title("Translation profiles", fontweight="bold")
                axes[i, 1].set_title("Rotation profiles", fontweight="bold")

            axes[i, 0].legend(loc="upper right", fontsize="small")
            axes[i, 1].legend(loc="upper right", fontsize="small")

        axes[-1, 0].set_xlabel(r"$x$ [m]")
        axes[-1, 1].set_xlabel(r"$x$ [m]")

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        plot_path = self.figure_output_dir / "deformation_convergence_overlay.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"✅ Saved: {plot_path}")

        if all_data:
            stacked = np.vstack(all_data)
            headers = ["job_id", "x", "ux", "uy", "uz", "theta_x", "theta_y", "theta_z"]

            if uy_analytical is not None:
                uy_interp = np.interp(stacked[:, 1], x_ref, uy_analytical * 1000)
                thetaz_interp = np.interp(stacked[:, 1], x_ref, np.degrees(thetaz_analytical))
                stacked = np.column_stack([stacked, uy_interp, thetaz_interp])
                headers += ["uy_analytical", "thetaz_analytical"]

            csv_path = self.figure_output_dir / "deformation_convergence_data.csv"
            np.savetxt(csv_path, stacked, delimiter=",", header=",".join(headers), comments="")
            print(f"✅ Saved: {csv_path}")


if __name__ == "__main__":
    VisualiseDeformationConvergence().process_convergence_plot()
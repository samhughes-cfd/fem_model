"""
Euler–Bernoulli vs FEM comparison
=================================
Plots for each selected job:

    • u_y   (vertical deflection)   ── millimetres
    • θ_z   (rotation about z)      ── degrees

The script relies on your mesh parser:

    pre_processing/parsing/mesh_parser.py   (function: parse_mesh)

Folder conventions (adjust only in helpers if they change)
----------------------------------------------------------
jobs/<job_id>/mesh.txt
post_processing/results/<job_id>_<timestamp>/primary_results/*_U_global.*
"""

from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# Add project root to sys.path so that the mesh_parser import always works
# ────────────────────────────────────────────────────────────────────────────────
def find_project_root(start: Path) -> Path:
    p = start
    while p != p.parent:
        if (p / "jobs").is_dir() and (p / "post_processing").is_dir():
            return p
        p = p.parent
    raise FileNotFoundError("Project root with 'jobs' and 'post_processing' not found")

SCRIPT_DIR   = Path(__file__).resolve().parent
ROOT         = find_project_root(SCRIPT_DIR)
sys.path.insert(0, str(ROOT))   # ensure project root is importable

from pre_processing.parsing.grid_parser import GridParser  # noqa: E402

SETTINGS_ROOT   = ROOT / "jobs"
RESULTS_ROOT    = ROOT / "post_processing" / "results"
PRIMARY_RESULTS = "primary_results"

# ────────────────────────────────────────────────────────────────────────────────
# Material / beam constants
# ────────────────────────────────────────────────────────────────────────────────
E   = 2.1e+11       # Pa
I_z = 2.08769e-06     # m⁴
F   = 500         # N
q_0 = 500         # N/m

# job → load description
job_to_loadtype = {
    "job_0001": "End Load",
    "job_0002": "Midpoint Load",
    "job_0003": "Quarterpoint Load",
    "job_0004": "Constant Distributed Load",
    "job_0005": "Quadratic Distributed Load",
    "job_0006": "Parabolic Distributed Load",
}

# ────────────────────────────────────────────────────────────────────────────────
# Discovery helpers
# ────────────────────────────────────────────────────────────────────────────────
def newest(paths: list[Path]) -> Path:
    return sorted(paths)[-1]

def settings_dir(job: str) -> Path:
    direct = SETTINGS_ROOT / job
    if direct.is_dir():
        return direct
    matches = [p for p in SETTINGS_ROOT.glob(f"{job}_*") if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"Settings directory not found for {job}")
    return newest(matches)

def mesh_file(job: str) -> Path:
    sd = settings_dir(job)
    candidate = sd / "mesh.txt"
    if candidate.is_file():
        return candidate
    patterns = ("*mesh*.csv", "*mesh*.txt", "*nodes*.csv", "*nodes*.txt")
    for pat in patterns:
        matches = list(sd.glob(pat))
        if matches:
            return newest(matches)
    raise FileNotFoundError(f"No mesh/nodes file in {sd}")

def results_dir(job: str) -> Path:
    matches = [p for p in RESULTS_ROOT.glob(f"{job}_*") if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"Results directory not found for {job}")
    return newest(matches)

def displacement_file(job: str) -> Path:
    rd = results_dir(job)
    file = rd / PRIMARY_RESULTS / "global" / "U_global.csv"
    if not file.is_file():
        raise FileNotFoundError(f"No U_global.csv file in {file.parent}")
    return file


# ────────────────────────────────────────────────────────────────────────────────
# File readers
# ────────────────────────────────────────────────────────────────────────────────
def read_grid_nodes_x(job: str) -> np.ndarray:
    """Parse grid.txt using GridParser and extract sorted unique X-coordinates."""
    settings = settings_dir(job)
    grid_file = settings / "grid.txt"
    if not grid_file.is_file():
        raise FileNotFoundError(f"No grid.txt found in {settings}")

    grid = GridParser(str(grid_file), str(settings)).parse()

    # Extract coordinates from grid["grid_dictionary"]["coordinates"]
    try:
        coords = grid["grid_dictionary"]["coordinates"]
    except (KeyError, TypeError):
        raise ValueError(f"Malformed grid dictionary for {job}")

    return np.sort(np.unique(coords[:, 0]))  # X-coordinate


# ────────────────────────────────────────────────────────────────────────────────
# Robust displacement-file reader  (handles 2-column DOF list, table, or 1-D vector)
# ────────────────────────────────────────────────────────────────────────────────
def read_dofs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts FEM results from a *_U_global.* file and returns:

        u_y_mm      — vertical displacements [mm]
        theta_z_deg — rotations about z [deg]

    Accepts three formats automatically:
    1) 2-column list    ->  index , value
    2) Wide table       ->  one row per node, ≥6 numeric DOF columns
    3) Flat 1-D vector  ->  [... ux, uy, uz, rx, ry, rz] * n_nodes
    """
    # ---------- try 2-column “index,value” list ----------
    data = np.genfromtxt(path,
                         delimiter=",",
                         skip_header=1,      # skip header if present
                         comments=None,
                         dtype=float,
                         invalid_raise=False)

    # If genfromtxt read something numeric, `data` is either (N,2) or (N,)
    if data.ndim == 2 and data.shape[1] == 2:
        values = data[:, 1]
        if values.size % 6 != 0:
            raise ValueError(
                f"{path.name}: DOF list length {values.size} not divisible by 6"
            )
        u_y_mm  = values[1::6] * 1_000.0
        rz_deg  = values[5::6] * 180.0 / np.pi
        return u_y_mm, rz_deg

    # ---------- try wide table (≥ 6 columns) ----------
    if data.ndim == 2 and data.shape[1] >= 6:
        # If first column looks like sequential node IDs, ignore it
        offset = 1 if np.allclose(data[:, 0], np.arange(data.shape[0])) else 0
        u_y_mm = data[:, 1 + offset] * 1_000.0
        rz_deg = data[:, 5 + offset] * 180.0 / np.pi
        return u_y_mm, rz_deg

    # ---------- fallback: flat 1-D vector ----------
    flat = np.genfromtxt(path,
                         delimiter=",",
                         comments=None,
                         dtype=float,
                         invalid_raise=False).flatten()
    flat = flat[np.isfinite(flat)]           # drop any NaNs
    if flat.size % 6 != 0:
        raise ValueError(
            f"{path.name}: cannot interpret file format – "
            f"2-column list and wide-table heuristics failed, "
            f"and {flat.size} numbers is not 6×n"
        )
    u_y_mm = flat[1::6] * 1_000.0
    rz_deg = flat[5::6] * 180.0 / np.pi
    return u_y_mm, rz_deg

# ────────────────────────────────────────────────────────────────────────────────
# Euler–Bernoulli theory
# ────────────────────────────────────────────────────────────────────────────────
def eb_deflection(x: np.ndarray, load: str) -> np.ndarray:
    L = x[-1]
    if load == "End Load":
        return (F * x**2) / (6 * E * I_z) * (3 * L - x)
    if load == "Midpoint Load":
        return (F * x**2) / (6 * E * I_z) * (3 * L - x)
    if load == "Quarterpoint Load":
        return (F * x**2) / (6 * E * I_z) * (3 * L - x)
    if load == "Constant Distributed Load":
        return (q_0 * x**2) / (24 * E * I_z) * (6 * L**2 - 4 * L * x + x**2)
    if load == "Quadratic Distributed Load":
        return (q_0 * x**4) / (30 * L**2) - (q_0 * x**5) / (10 * L) + (q_0 * x**6) / 12
    if load == "Parabolic Distributed Load":
        return (q_0 * x**5) / (20 * L**2) - (q_0 * x**6) / (30 * L**3) - (q_0 * x**4) / 12
    return np.zeros_like(x)

def eb_slope_deg(x: np.ndarray, load: str) -> np.ndarray:
    slope_rad = np.gradient(eb_deflection(x, load), x)
    return slope_rad * 180.0 / np.pi

# ────────────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────────────

def plot_jobs(jobs: list[str]) -> None:
    """
    Make a single figure with 1×2 sub-plots:
        • left  : u_y  [mm]   vs x
        • right : θ_z [deg]   vs x
    Every selected job is plotted on the same pair of axes.
    """
    # ---------- gather all data -------------------------------------------------
    artefacts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, str]] = {}
    for job in jobs:
        try:
            x_coords       = read_grid_nodes_x(job)
            uy_mm, rz_deg  = read_dofs(displacement_file(job))
            load           = job_to_loadtype.get(job, "Unknown")
            artefacts[job] = (x_coords, uy_mm, rz_deg, load)
        except (FileNotFoundError, ValueError) as e:
            print(f"[WARN] {e}")

    if not artefacts:
        sys.exit("Nothing to plot – no valid jobs found.")

    # ---------- prepare figure --------------------------------------------------
    fig, (ax_u, ax_th) = plt.subplots(
        1, 2, figsize=(12, 5), sharex=False, constrained_layout=True
    )

    # Give each job a distinct colour cycle automatically
    for job_idx, (job, (x, uy_mm, rz_deg, load)) in enumerate(artefacts.items()):
        # Theory
        eb_mm  = eb_deflection(x, load) * 1_000.0
        eb_deg = eb_slope_deg(x, load)

        if load != "Parabolic Distributed Load":
            eb_mm  *= -1.0
            eb_deg *= -1.0

        # ---------- u_y subplot (left) ----------
        ax_u.plot(x, uy_mm,
                  marker="o", linestyle="-",  label=f"{job} FEM")
        ax_u.plot(x, eb_mm,
                  linestyle="--",            label=f"{job} EB")

        # ---------- θ_z subplot (right) ----------
        ax_th.plot(x, rz_deg,
                   marker="s", linestyle="-",  label=f"{job} FEM")
        ax_th.plot(x, eb_deg,
                   linestyle=":",             label=f"{job} EB")

    # ---------- cosmetics -------------------------------------------------------
    ax_u.set_title("$u_y$ comparison")
    ax_u.set_xlabel("x [m]")
    ax_u.set_ylabel("$u_y$ [mm]")
    ax_u.grid(True, linestyle="--", alpha=0.7)
    ax_u.legend()

    ax_th.set_title("$\\theta_z$ comparison")
    ax_th.set_xlabel("x [m]")
    ax_th.set_ylabel("$\\theta_z$ [deg]")
    ax_th.grid(True, linestyle="--", alpha=0.7)
    ax_th.legend()

    plt.show()

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Edit this list to choose which jobs to plot
    JOBS = ["job_0001"]

    plot_jobs(JOBS)
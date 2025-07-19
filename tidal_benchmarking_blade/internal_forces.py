"""
Blade internal-action diagrams for flapwise, edgewise and torsion
-----------------------------------------------------------------

* Reads distributed loads (N /m, Nm /m) from `load_profiles/*.csv`
* Produces a 3 × 3 grid:

      ┌──────────┬──────────┬──────────┐
      │ flapwise │ edgewise │ torsion  │
      ├──────────┼──────────┼──────────┤
      │  Fₙy     │  Fₙz     │   Mₓ     │  ← distributed loads
      │  Vᵧ      │  Vz      │  T(x)   │  ← internal resultants
      │  Mz      │  My      │   —      │  ← bending moments
      └──────────┴──────────┴──────────┘

  (T(x) sits on the middle row and keeps its own x-axis ticks & label.)

* Outputs
      outputs/blade_internal_actions.png
      outputs/<TSR>_processed.csv  (one per TSR)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from labellines import labelLines

# ───────── configuration ─────────
R          = 0.80  # full rotor radius in meters
L          = 0.70  # blade span length (from root to tip) in meters
offset     = R - L  # accounts for physical offset of blade root (0.10 m)
TSR_NAMES  = ["TSR4", "TSR5", "TSR6", "TSR7", "TSR8"]

COLORS = {
    "load":    "#4F81BD",   # Fy, Fz, Mx
    "shear":   "#9BBB59",   # Vy, Vz
    "bending": "#C0504D",   # Mz, My
    "torsion": "#8064A2",   # T(x)
}

# ───── file-system setup ─────
SCRIPT_DIR = Path(__file__).resolve().parent
LOAD_DIR   = SCRIPT_DIR / "load_profiles"
OUT_DIR    = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not LOAD_DIR.is_dir():
    sys.exit(f"[ERROR] Expected load profiles in: {LOAD_DIR}")

# ───── helper: cumulative trapezoid ─────
def cumtrapz_uniform(y: np.ndarray, dx: float) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum((y[:-1] + y[1:]) * 0.5 * dx)))

# ───────── main routine ─────────
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(13, 10), sharex="col")

# ─── NEW subplot titles ───
axs[0, 0].set_title("Flapwise")
axs[0, 1].set_title("Edgewise")
axs[0, 2].set_title("Torsion")

for tsr in TSR_NAMES:
    csv_path = LOAD_DIR / f"{tsr}.csv"
    if not csv_path.is_file():
        print(f"[WARNING] Skipping {tsr}: file not found → {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip("[]")

    x   = df["x"].to_numpy()  # physical x along blade span (in meters)
    dx  = float(np.mean(np.diff(x)))
    rR  = (x + offset) / R    # correct r/R offset

    # distributed loads  (signs per your convention)
    qy  =  df["F_y"].to_numpy()          # N/m
    qz  = -df["F_z"].to_numpy()          # N/m
    mx  =  df["M_x"].to_numpy()          # Nm/m

    # internal resultants
    Vy  =  cumtrapz_uniform(qy[::-1], dx)[::-1]
    Vz  =  cumtrapz_uniform(qz[::-1], dx)[::-1]
    Mz  = -cumtrapz_uniform(Vy[::-1], dx)[::-1]
    My  = -cumtrapz_uniform(Vz[::-1], dx)[::-1]
    T   =  cumtrapz_uniform(mx[::-1], dx)[::-1]

    # ── plotting ──
    # column 0 – flapwise
    axs[0, 0].plot(rR, qy, label=tsr, color=COLORS["load"])
    axs[1, 0].plot(rR, Vy, label=tsr, color=COLORS["shear"])
    axs[2, 0].plot(rR, Mz, label=tsr, color=COLORS["bending"])

    # column 1 – edgewise
    axs[0, 1].plot(rR, qz, label=tsr, color=COLORS["load"])
    axs[1, 1].plot(rR, Vz, label=tsr, color=COLORS["shear"])
    axs[2, 1].plot(rR, My, label=tsr, color=COLORS["bending"])

    # column 2 – torsion
    axs[0, 2].plot(rR, mx, label=tsr, color=COLORS["load"])
    axs[1, 2].plot(rR, T,  label=tsr, color=COLORS["torsion"])

    # save processed CSV
    pd.DataFrame({
        "r/R": rR,
        "f_y [N/m]": qy, "V_y [N]": Vy, "M_z [Nm]": Mz,
        "f_z [N/m]": qz, "V_z [N]": Vz, "M_y [Nm]": My,
        "m_x [Nm/m]": mx, "T [Nm]": T
    }).to_csv(OUT_DIR / f"{tsr}_processed.csv", index=False)

# ───── labels & layout ─────
ylabels = [
    ("$f_y$ [N/m]", "$f_z$ [N/m]", "$m_x$ [Nm/m]"),
    ("$V_y(x)$ [N]", "$V_z(x)$ [N]", "$T(x)$ [Nm]"),
    ("$M_z(x)$ [Nm]", "$M_y(x)$ [Nm]", ""),
]

for r in range(3):
    for c in range(3):
        label = ylabels[r][c]
        if label == "":
            axs[r, c].axis("off")
            continue
        axs[r, c].set_ylabel(label)
        axs[r, c].grid(True, zorder=0)
        axs[r, c].set_xlim(-0.02, 1.02)
        if (r < 2) and not (r == 1 and c == 2):   # keep ticks for T(x) pane
            axs[r, c].tick_params(labelbottom=False)

# x-labels
axs[2, 0].set_xlabel("r / R")
axs[2, 1].set_xlabel("r / R")
axs[1, 2].set_xlabel("r / R")
axs[1, 2].tick_params(labelbottom=True)

# ───── vertical lines & labels ─────
for ax in axs.flat:
    if ax.get_visible():
        # Skip bottom-right panel [2, 2]
        if ax == axs[2, 2]:
            continue
        ax.axvline(x=0.0, linestyle='--', color='black', linewidth=1.0, zorder=2)
        ax.axvline(x=0.125, linestyle='--', color='black', linewidth=1.0, zorder=2)
        ax.axvline(x=1.0, linestyle='--', color='black', linewidth=1.0, zorder=2)
        if ax.lines:
            labelLines(ax.get_lines(), zorder=3)

# ───── finalize and save ─────
fig.suptitle("Internal Force Equilibrium Diagram – TSR4 – TSR8", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])

png_path = OUT_DIR / "blade_internal_actions.png"
fig.savefig(png_path, dpi=300)
plt.show()

print(f"[DONE] Figures and CSVs written to: {OUT_DIR}")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from labellines import labelLines

# ───────── Configuration ─────────
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_PNG = INPUT_DIR / "worst_case_internal_actions.png"

COLORS = {
    "Mz": "#C0504D",  # red-ish
    "My": "#C0504D",  # red-ish
    "T":  "#8064A2",  # purple-ish
}

# ───────── Initialize ─────────
worst_curves = {
    "Mz": {"tsr": None, "rR": None, "values": None, "max_val": -np.inf},
    "My": {"tsr": None, "rR": None, "values": None, "max_val": -np.inf},
    "T":  {"tsr": None, "rR": None, "values": None, "max_val": -np.inf},
}

# ───────── Read All Processed Files ─────────
for csv_file in INPUT_DIR.glob("TSR*_processed.csv"):
    tsr_name = csv_file.stem.split("_")[0]
    df = pd.read_csv(csv_file)

    rR = df["r/R"].to_numpy()
    Mz = df["M_z [Nm]"].to_numpy()
    My = df["M_y [Nm]"].to_numpy()
    T  = df["T [Nm]"].to_numpy()

    for key, series in zip(["Mz", "My", "T"], [Mz, My, T]):
        max_val = np.max(np.abs(series))
        if max_val > worst_curves[key]["max_val"]:
            worst_curves[key] = {
                "tsr": tsr_name,
                "rR": rR,
                "values": series,
                "max_val": max_val
            }

# ───────── Plotting ─────────
fig, ax = plt.subplots(figsize=(10, 6))

# Prepare curves
rR = worst_curves["Mz"]["rR"]
zero = np.zeros_like(rR)
curves = {
    "Mz": worst_curves["Mz"]["values"],
    "My": worst_curves["My"]["values"],
    "T":  worst_curves["T"]["values"],
}

# ───────── Generalized Nested Fill ─────────
def fill_nested_segment(ax, x, y_stack, curve_order, color_map, alpha=0.25):
    for i in range(len(y_stack) - 1):
        lower = np.array(y_stack[i])
        upper = np.array(y_stack[i + 1])
        key = curve_order[i]
        mask_up = upper >= lower
        mask_dn = upper < lower
        ax.fill_between(x, lower, upper, where=mask_up, color=color_map[key], alpha=alpha, linewidth=0, zorder=2)
        ax.fill_between(x, lower, upper, where=mask_dn, color=color_map[key], alpha=alpha, linewidth=0, zorder=2)

# Fill by segment
curve_order = ["T", "My", "Mz"]
for i in range(len(rR) - 1):
    xseg = [rR[i], rR[i + 1]]
    y_stack = [
        [0, 0],
        [curves["T"][i], curves["T"][i + 1]],
        [curves["My"][i], curves["My"][i + 1]],
        [curves["Mz"][i], curves["Mz"][i + 1]],
    ]
    fill_nested_segment(ax, xseg, y_stack, curve_order, COLORS)

# ───────── Plot Curves ─────────
for key in ["Mz", "My", "T"]:
    data = worst_curves[key]
    ax.plot(data["rR"], data["values"], label=f"${key}(x)$ – {data['tsr']}", color=COLORS[key], zorder=5)

# ───────── Add Markers at Extremes ─────────
# Max positive My
My_vals = curves["My"]
idx_max_My = np.argmax(My_vals)
ax.plot(
    rR[idx_max_My], My_vals[idx_max_My],
    marker='o', markersize=5, color=COLORS["My"], zorder=9,
    label="_nolegend_"
)

# Max positive T
T_vals = curves["T"]
idx_max_T = np.argmax(T_vals)
ax.plot(
    rR[idx_max_T], T_vals[idx_max_T],
    marker='o', markersize=5, color=COLORS["T"], zorder=9,
    label="_nolegend_"
)

# Max negative Mz
Mz_vals = curves["Mz"]
idx_min_Mz = np.argmin(Mz_vals)
ax.plot(
    rR[idx_min_Mz], Mz_vals[idx_min_Mz],
    marker='o', markersize=5, color=COLORS["Mz"], zorder=9,
    label="_nolegend_"
)


# ───────── Style ─────────
for xpos in [0.0, 0.125, 1.0]:
    ax.axvline(x=xpos, linestyle='--', color='black', linewidth=1.0, zorder=1)

labelLines(ax.get_lines(), zorder=6)
ax.set_xlim(-0.02, 1.02)
ax.set_xlabel("r / R [-]")
ax.set_ylabel("BM/TT Diagram Space [Nm]")
ax.set_title("Extreme Case of Internal Action Space TSR4–TSR8")
ax.grid(True)
fig.tight_layout()

fig.savefig(OUTPUT_PNG, dpi=300)
plt.show()
print(f"[DONE] Worst-case plot saved to: {OUTPUT_PNG}")

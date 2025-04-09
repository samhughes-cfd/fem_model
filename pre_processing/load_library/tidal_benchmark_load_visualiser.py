# pre_processing\load_library\tidal_benchmark_load_visualiser.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from labellines import labelLines
import pandas as pd
import glob
import logging

# -----------------------------------------
# ✅ Add project root to sys.path
# -----------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# -----------------------------------------
# ✅ Import custom parser
# -----------------------------------------
from pre_processing.parsing.distributed_load_parser import parse_distributed_load

# -----------------------------------------
# ✅ Logging Setup
# -----------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# -----------------------------------------
# ✅ File Discovery
# -----------------------------------------
data_directory = os.path.abspath("pre_processing/load_library/load_profiles/tidal_benchmark/")
file_pattern = os.path.join(data_directory, "load_profile_TSR*_*.txt")
files = glob.glob(file_pattern)

if not files:
    logging.error("No files found. Check the directory path: %s", data_directory)
    sys.exit(1)

# -----------------------------------------
# ✅ TSR Extraction Function
# -----------------------------------------
def extract_tsr(filename):
    try:
        return int(filename.split("TSR")[-1].split("_")[0])
    except Exception as e:
        logging.warning(f"Could not extract TSR from filename {filename}: {e}")
        return None

# -----------------------------------------
# ✅ Load and Parse All Files
# -----------------------------------------
data_dict = {}
for file in files:
    tsr = extract_tsr(file)
    if tsr is None:
        continue
    try:
        data = parse_distributed_load(file)
        df = pd.DataFrame(data, columns=["x", "y", "z", "F_x", "F_y", "F_z", "M_x", "M_y", "M_z"])
        data_dict[tsr] = df
    except Exception as e:
        logging.error(f"Failed to parse data from file {file}: {e}")

sorted_tsrs = sorted(data_dict.keys())

# -----------------------------------------
# ✅ Plotting Function
# -----------------------------------------
def plot_load_profiles():
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)

    force_components = ['F_x', 'F_y', 'F_z']
    moment_components = ['M_x', 'M_y', 'M_z']
    force_titles = ["Force in radial direction", "Force in flapwise direction", "Force in edgewise direction"]
    moment_titles = ["Moment around radial axis", "Moment around flapwise axis", "Moment around edgewise axis"]
    y_labels = [r"$F_x$ (N)", r"$F_y$ (N)", r"$F_z$ (N)", r"$M_x$ (Nm)", r"$M_y$ (Nm)", r"$M_z$ (Nm)"]

    for i in range(3):
        for tsr in sorted_tsrs:
            df = data_dict[tsr]
            axes[i, 0].plot(df['x'], df[force_components[i]], label=f'TSR {tsr}')
            axes[i, 1].plot(df['x'], df[moment_components[i]], label=f'TSR {tsr}')

        axes[i, 0].set_title(force_titles[i])
        axes[i, 0].set_ylabel(y_labels[i])
        axes[i, 1].set_title(moment_titles[i])
        axes[i, 1].set_ylabel(y_labels[i + 3])

        axes[i, 0].text(0.05, 0.95, r"$TSR = \Omega R (U_\infty)^{-1}$",
                       transform=axes[i, 0].transAxes, fontsize=10, verticalalignment='top')
        axes[i, 1].text(0.05, 0.95, r"$TSR = \Omega R (U_\infty)^{-1}$",
                       transform=axes[i, 1].transAxes, fontsize=10, verticalalignment='top')

        labelLines(axes[i, 0].get_lines(), zorder=2.5)
        labelLines(axes[i, 1].get_lines(), zorder=2.5)

    # Dynamic x-limits
    all_x_values = np.concatenate([df['x'].values for df in data_dict.values()])
    xmin, xmax = min(all_x_values), max(all_x_values)
    margin_x = 0.05 * (xmax - xmin)
    for ax in axes[:, 0]:
        ax.set_xlim(xmin - margin_x, xmax + margin_x)
    for ax in axes[:, 1]:
        ax.set_xlim(xmin - margin_x, xmax + margin_x)

    axes[-1, 0].set_xlabel(r"Radial Position $x$ (m)")
    axes[-1, 1].set_xlabel(r"Radial Position $x$ (m)")

    plt.tight_layout()
    plt.show()

# -----------------------------------------
# ✅ Run Plot
# -----------------------------------------
plot_load_profiles()
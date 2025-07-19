import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Absolute path to data directory
data_dir = os.path.abspath("tidal_benchmarking_blade/load_profiles")

# Safely extract TSR value from filenames like "TSR4.csv"
def extract_tsr(filename):
    try:
        base = os.path.basename(filename)
        if base.lower().startswith("tsr") and base.lower().endswith(".csv"):
            return float(base[3:].split(".")[0])
    except (IndexError, ValueError):
        pass
    return float('inf')  # Push invalid files to end

# Compute r/R from x
def compute_r_over_R(x):
    return (x - 0.1) / 0.7

# Define spanwise r/R targets and convert to x
r_over_R_targets = np.linspace(0.125, 1.0, 10)
x_targets = 0.1 + 0.7 * r_over_R_targets

# Initialize storage
TSRs = []
Fy_matrix = []
Fz_matrix = []
Mx_matrix = []

# Filter and sort filenames by TSR value
filenames = [f for f in os.listdir(data_dir) if f.lower().startswith("tsr") and f.lower().endswith(".csv")]
filenames = sorted(filenames, key=extract_tsr)

# Read and process each file
for filename in filenames:
    tsr = extract_tsr(filename)
    if tsr == float('inf'):
        continue

    TSRs.append(tsr)
    file_path = os.path.join(data_dir, filename)
    df = pd.read_csv(file_path)

    x_data = df["[x]"].to_numpy()
    Fy_data = df["[F_y]"].to_numpy()
    Fz_data = df["[F_z]"].to_numpy()
    Mx_data = df["[M_x]"].to_numpy()

    Fy_slice = []
    Fz_slice = []
    Mx_slice = []
    for xt in x_targets:
        idx = np.argmin(np.abs(x_data - xt))
        Fy_slice.append(Fy_data[idx])
        Fz_slice.append(Fz_data[idx])
        Mx_slice.append(Mx_data[idx])

    Fy_matrix.append(Fy_slice)
    Fz_matrix.append(Fz_slice)
    Mx_matrix.append(Mx_slice)

# Convert to arrays
TSRs = np.array(TSRs)
Fy_matrix = np.array(Fy_matrix)
Fz_matrix = np.array(Fz_matrix)
Mx_matrix = np.array(Mx_matrix)

# Plotting function
def plot_quantity_vs_tsr(matrix, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    for i, rR in enumerate(r_over_R_targets):
        plt.plot(TSRs, matrix[:, i], label=f"r/R = {rR:.3f}", marker='o')
    plt.xlabel("TSR")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Span Location")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Save plots
plot_quantity_vs_tsr(Fy_matrix, "F_y [N]", "Spanwise F_y vs TSR", "Fy_vs_TSR.png")
plot_quantity_vs_tsr(Fz_matrix, "F_z [N]", "Spanwise F_z vs TSR", "Fz_vs_TSR.png")
plot_quantity_vs_tsr(Mx_matrix, "M_x [Nm]", "Spanwise M_x vs TSR", "Mx_vs_TSR.png")
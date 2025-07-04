import pandas as pd
import matplotlib.pyplot as plt
from labellines import labelLines
import numpy as np
import os

# Constants
R = 0.7
tsr_names = ["TSR4", "TSR5", "TSR6", "TSR7", "TSR8"]
base_dir = r"\\mull.sms.ed.ac.uk\home\s1834431\Win7\Desktop\fem_model\tidal_benchmarking_blade"
load_profile_dir = r"\\mull.sms.ed.ac.uk\home\s1834431\Win7\Desktop\fem_model\tidal_benchmarking_blade\load_profiles"
output_dir = os.path.join(base_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

# Colors
load_color = "#4F81BD"
shear_color = "#9BBB59"
moment_color = "#C0504D"

# Create 3 vertically stacked subplots with shared x-axis
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

for tsr in tsr_names:
    file_path = os.path.join(load_profile_dir, f"{tsr}.csv")
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip("[]")

    x = df["x"]
    r_by_R = x / R
    fy = -df["F_y"]  # downward force assumed negative

    dx = x.diff().fillna(0)
    V = fy[::-1].cumsum()[::-1] * dx.mean()
    Mz = -V[::-1].cumsum()[::-1] * dx.mean()

    # Plotting
    axs[0].plot(r_by_R, fy, label=tsr, color=load_color)
    axs[1].plot(r_by_R, V, label=tsr, color=shear_color)
    axs[2].plot(r_by_R, Mz, label=tsr, color=moment_color)

    # Write processed CSV
    output_df = pd.DataFrame({
        "r/R": r_by_R,
        "Fy": fy,
        "V": V,
        "Mz": Mz
    })
    csv_out_path = os.path.join(output_dir, f"{tsr}_output.csv")
    output_df.to_csv(csv_out_path, index=False)

# Titles and y-labels
axs[0].set_ylabel("$F_y$ [N]")
axs[1].set_ylabel("$V(x)$ [N]")
axs[2].set_ylabel("$M_z(x)$ [Nm]")
axs[2].set_xlabel("r / R")

# Hide x-tick labels for top two plots
axs[0].tick_params(labelbottom=False)
axs[1].tick_params(labelbottom=False)

# Grid and label lines
for ax in axs:
    ax.grid(True)
    labelLines(ax.get_lines(), zorder=2.5)

# Layout and save
fig.suptitle("Tidal Benchmarking Blade for TSR4–TSR8", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
output_path = os.path.join(base_dir, "tidal_benchmarking_blade_TSR4-TSR8.png")
plt.savefig(output_path, dpi=300)
plt.show()
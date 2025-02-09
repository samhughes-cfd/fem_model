import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from labellines import labelLines
import pandas as pd
import glob

# Define the correct absolute path to the files
data_directory = os.path.abspath("pre_processing/load_library/load_profiles/tidal_benchmark/")
file_pattern = os.path.join(data_directory, "load_profile_TSR*_*.txt")

# Find all matching files
files = glob.glob(file_pattern)

# Debugging: Print the found files
if not files:
    print("No files found. Check the directory path:", data_directory)
    sys.exit(1)

def extract_tsr(filename):
    """Extract TSR value from filename."""
    return int(filename.split("TSR")[-1].split("_")[0])

# Read data from all files
data_dict = {}
for file in files:
    tsr = extract_tsr(file)
    df = pd.read_csv(file, delim_whitespace=True, skiprows=1, names=["x", "y", "z", "F_x", "F_y", "F_z", "M_x", "M_y", "M_z"])
    
    # Shift all x-values by 0.1m to account for hub radius
    df["x"] += 0.1  
    data_dict[tsr] = df

# Sort TSRs for ordered plotting
sorted_tsrs = sorted(data_dict.keys())

# Function to plot forces and moments in a 2-column format
def plot_load_profiles(xlabel_size=12, ylabel_size=12):
    """
    Plots force and moment distributions along the blade span.

    Parameters:
    - xlabel_size (int): Font size for x-axis labels.
    - ylabel_size (int): Font size for y-axis labels.
    """

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
        axes[i, 0].set_ylabel(y_labels[i], fontsize=ylabel_size)
        axes[i, 1].set_title(moment_titles[i])
        axes[i, 1].set_ylabel(y_labels[i + 3], fontsize=ylabel_size)
        
        # Add TSR definition in the upper left corner of each subplot
        axes[i, 0].text(0.05, 0.95, r"$TSR = \Omega R (U_\infty)^{-1}$", transform=axes[i, 0].transAxes, fontsize=10, verticalalignment='top')
        axes[i, 1].text(0.05, 0.95, r"$TSR = \Omega R (U_\infty)^{-1}$", transform=axes[i, 1].transAxes, fontsize=10, verticalalignment='top')
        
        # Inline labels
        labelLines(axes[i, 0].get_lines(), zorder=2.5)
        labelLines(axes[i, 1].get_lines(), zorder=2.5)
    
    # Explicitly set x-axis limits to ensure full span from 0 to 0.8m
    for ax in axes[:, 0]:
        ax.set_xlim(0, 0.8)
    for ax in axes[:, 1]:
        ax.set_xlim(0, 0.8)
    
    # X-axis labels with user-defined font size
    axes[-1, 0].set_xlabel(r"Position $r$ (m)", fontsize=xlabel_size)
    axes[-1, 1].set_xlabel(r"Position $r$ (m)", fontsize=xlabel_size)
    
    plt.tight_layout()
    plt.show()

# Plot Forces and Moments in 2-column layout with adjustable label sizes
plot_load_profiles(xlabel_size=14, ylabel_size=14)  # Example usage with larger font sizes
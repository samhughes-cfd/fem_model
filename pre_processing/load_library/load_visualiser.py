# load_library/schemes/visualize_load.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from labellines import labelLines
import pandas as pd
import glob

# Define the file pattern and extract all matching files
file_pattern = "load_profile_TSR*_*.txt"
files = glob.glob(file_pattern)

def extract_tsr(filename):
    """Extract TSR value from filename."""
    return int(filename.split("TSR")[-1].split("_")[0])

# Read data from all files
data_dict = {}
for file in files:
    tsr = extract_tsr(file)
    df = pd.read_csv(file, delim_whitespace=True)
    data_dict[tsr] = df

# Sort TSRs for ordered plotting
sorted_tsrs = sorted(data_dict.keys())

# Function to plot forces and moments with inline labels
def plot_load_profiles(title, components):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    fig.suptitle(title, fontsize=14)
    
    for i, comp in enumerate(components):
        for tsr in sorted_tsrs:
            df = data_dict[tsr]
            axes[i].plot(df['x'], df[comp], label=f'TSR {tsr}')
        
        axes[i].set_title(f'{comp} vs x')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel(comp)
        labelLines(axes[i].get_lines(), zorder=2.5)
    
    plt.tight_layout()
    plt.show()

# Plot Forces
plot_load_profiles("Force Components vs x", ['F_x', 'F_y', 'F_z'])

# Plot Moments
plot_load_profiles("Moment Components vs x", ['M_x', 'M_y', 'M_z'])
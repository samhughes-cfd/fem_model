# post_processing\graphical_visualisers\load\load_visualisation.py

import os
import sys
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Set up paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)

# Use your custom parsers and mesh parser
from pre_processing.parsing.mesh_parser import parse_mesh
from pre_processing.parsing.point_load_parser import parse_point_load
from pre_processing.parsing.distributed_load_parser import parse_distributed_load

# Set output directory for figures similar to deformation plots
FIGURE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "load_plots")
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

def plot_load_style(load_data, job_id, load_type, save_path):
    if load_data is None or load_data.shape[1] != 9:
        print(f"Invalid load data shape for job_{job_id}: {load_data.shape if load_data is not None else 'None'}")
        return

    x_vals = load_data[:, 0]
    forces = load_data[:, 3:6]
    moments = load_data[:, 6:9]
    labels = [r"$F_x$", r"$F_y$", r"$F_z$", r"$M_x$", r"$M_y$", r"$M_z$"]

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    fig.suptitle(f"{load_type.capitalize()} Load - job_{job_id}", fontsize=16, fontweight='bold')
    color = "blue"

    for i in range(3):
        ax_f = axes[i, 0]
        ax_m = axes[i, 1]

        # Plot forces
        if load_type == "distributed":
            ax_f.plot(x_vals, forces[:, i], color=color, linewidth=2)
            ax_f.fill_between(x_vals, forces[:, i], 0, color=color, alpha=0.25)
        else:
            # For point loads, draw vertical lines with an arrowhead marker where nonzero.
            for x, y in zip(x_vals, forces[:, i]):
                if y != 0:
                    ax_f.plot([x, x], [0, y], color=color, linewidth=2)
                    ax_f.plot(x, y, marker="v", color=color, markersize=8)
        
        # Plot moments
        if load_type == "distributed":
            ax_m.plot(x_vals, moments[:, i], color=color, linewidth=2)
            ax_m.fill_between(x_vals, moments[:, i], 0, color=color, alpha=0.25)
        else:
            for x, y in zip(x_vals, moments[:, i]):
                if y != 0:
                    ax_m.plot([x, x], [0, y], color=color, linewidth=2)
                    ax_m.plot(x, y, marker="v", color=color, markersize=8)

        ax_f.set_ylabel(labels[i], fontsize=12)
        ax_m.set_ylabel(labels[i + 3], fontsize=12)

        ax_f.axhline(0, color='k', linestyle='--', linewidth=0.75)
        ax_m.axhline(0, color='k', linestyle='--', linewidth=0.75)
        ax_f.grid(True, linestyle='--', alpha=0.6)
        ax_m.grid(True, linestyle='--', alpha=0.6)

        if i == 0:
            ax_f.set_title("Forces", fontsize=12, fontweight='bold')
            ax_m.set_title("Moments", fontsize=12, fontweight='bold')

    axes[-1, 0].set_xlabel(r"$x$ [m]", fontsize=12)
    axes[-1, 1].set_xlabel(r"$x$ [m]", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def process_all_loads():
    jobs_dir = os.path.join(PROJECT_ROOT, "jobs")
    job_dirs = sorted(glob.glob(os.path.join(jobs_dir, "job_*")))

    for job_path in job_dirs:
        job_id_match = re.search(r"job_(\d+)", job_path)
        if not job_id_match:
            continue
        job_id = job_id_match.group(1)

        mesh_path = os.path.join(job_path, "mesh.txt")
        if not os.path.exists(mesh_path):
            continue

        mesh_dict = parse_mesh(mesh_path)
        if 'node_coordinates' not in mesh_dict:
            continue

        # Loop over both load types with their corresponding parser functions.
        for load_type, parser_func in {
            "point": parse_point_load,
            "distributed": parse_distributed_load
        }.items():
            load_file = os.path.join(job_path, f"{load_type}_load.txt")
            if os.path.exists(load_file):
                print(f"Processing {load_type} load for job_{job_id}")
                try:
                    load_data = parser_func(load_file)
                    # Obtain a timestamp from the modification time of the load file
                    timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(load_file)).strftime("%Y-%m-%d_%H-%M-%S")
                    fig_name = f"load_job_{job_id}_{timestamp}.png"
                    fig_path = os.path.join(FIGURE_OUTPUT_DIR, fig_name)
                    plot_load_style(load_data, job_id, load_type, fig_path)
                except Exception as e:
                    print(f"Failed to parse {load_type} load for job_{job_id}: {e}")

if __name__ == "__main__":
    process_all_loads()
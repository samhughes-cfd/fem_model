# post_processing\graphical_visualisers\deformation\deformation_visualisation.py

import os
import sys
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)

from pre_processing.parsing.mesh_parser import parse_mesh

FIGURE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "deformation_plots")
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

def visualize_deformation(U, node_positions, scale=1.0, title_suffix="", save_path=None):
    if U.shape[1] != 6:
        raise ValueError("Input U must have shape (n_nodes, 6)")

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    title = r"Raw $U_g (x)$"
    if title_suffix:
        title += f" - {title_suffix}"
    fig.suptitle(title, fontsize=16, fontweight='bold')

    color = "#4F81BD"
    deformation_pairs = [
        (U[:, 0] * 1000 * scale, r"$u_x(x)$ [mm]", U[:, 3], r"$\theta_x(x)$ [°]"),
        (U[:, 1] * 1000 * scale, r"$u_y(x)$ [mm]", U[:, 5], r"$\theta_z(x)$ [°]"),
        (U[:, 2] * 1000 * scale, r"$u_z(x)$ [mm]", U[:, 4], r"$\theta_y(x)$ [°]"),
    ]

    for i, (ax_left, ax_right, (disp, disp_label, rot, rot_label)) in enumerate(zip(axes[:, 0], axes[:, 1], deformation_pairs)):
        # Displacement plot
        ax_left.plot(node_positions, disp, color=color, linewidth=2, marker='o',
                     markerfacecolor=color, markeredgecolor=color)
        ax_left.axhline(0, color='k', linestyle='--', linewidth=0.75)
        ax_left.grid(True, linestyle='--', alpha=0.6)
        ax_left.set_ylabel(disp_label, fontsize=12)

        # Rotation plot
        rot_deg = np.degrees(rot) * scale
        ax_right.plot(node_positions, rot_deg, color=color, linewidth=2, marker='o',
                      markerfacecolor=color, markeredgecolor=color)
        ax_right.axhline(0, color='k', linestyle='--', linewidth=0.75)
        ax_right.grid(True, linestyle='--', alpha=0.6)
        ax_right.set_ylabel(rot_label, fontsize=12)

        if i == 0:
            ax_left.set_title("Translation profiles", fontsize=12, fontweight='bold')
            ax_right.set_title("Rotation profiles", fontsize=12, fontweight='bold')

    axes[-1, 0].set_xlabel(r"$x$ [m]", fontsize=12)
    axes[-1, 1].set_xlabel(r"$x$ [m]", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()

def read_deformation_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = np.loadtxt(file, comments='#')
            return data.reshape(-1, 6)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def process_deformation_files():
    base_dir = os.path.join(PROJECT_ROOT, "post_processing", "results")
    mesh_dir = os.path.join(PROJECT_ROOT, "jobs")
    search_pattern = os.path.join(base_dir, "job_*_*", "primary_results", "job_*_static_global_U_global_*.txt")
    files = sorted(glob.glob(search_pattern))

    if not files:
        print("No deformation files found.")
        return

    for file_path in files:
        match = re.search(r"job_(\d+)_([0-9\-]+_[0-9\-]+)", file_path)
        if not match:
            print(f"Skipping unrecognized file format: {file_path}")
            continue

        job_id, timestamp = match.groups()
        mesh_path = os.path.join(mesh_dir, f"job_{job_id}", "mesh.txt")

        print(f"Processing: {file_path}")
        U = read_deformation_file(file_path)
        mesh_dictionary = parse_mesh(mesh_path)

        if U is not None and mesh_dictionary is not None and 'node_coordinates' in mesh_dictionary:
            node_positions = mesh_dictionary['node_coordinates'][:, 0]
            title_suffix = f"job_{job_id}_{timestamp}"
            fig_name = f"deformation_job_{job_id}_{timestamp}.png"
            fig_path = os.path.join(FIGURE_OUTPUT_DIR, fig_name)
            visualize_deformation(U, node_positions, scale=1.0, title_suffix=title_suffix, save_path=fig_path)

if __name__ == "__main__":
    process_deformation_files()
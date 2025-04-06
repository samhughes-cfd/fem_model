# post_processing\graphical_visualisers\deflection_visualisation.py

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from pre_processing.parsing.mesh_parser import parse_mesh

def visualize_deformation(U, node_positions, scale=1.0, title_suffix=""):
    """
    Visualizes the six components of a deformation vector U = [u_x, u_y, u_z, theta_x, theta_y, theta_z]
    along the length of the element with aesthetics similar to Roark's formulas visualization.
    :param U: A numpy array of shape (n_nodes, 6) containing deformations at each node.
    :param node_positions: Numpy array of node positions along x-axis.
    :param scale: Scaling factor for visualization clarity.
    :param title_suffix: Optional string to append to the plot title (e.g., job ID or timestamp).
    """
    if U.shape[1] != 6:
        raise ValueError("Input U must have shape (n_nodes, 6) where each row contains [u_x, u_y, u_z, theta_x, theta_y, theta_z]")

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    title = r"Raw $U_g (x)$"
    if title_suffix:
        title += f" - {title_suffix}"
    fig.suptitle(title, fontsize=16, fontweight='bold')

    color = "#4F81BD"  # Consistent blue color for all plots

    deformation_pairs = [
        (U[:, 0], r"$u_x(x) \,[\mathrm{mm}]$", U[:, 3], r"$\theta_x(x) \,[\degree]$"),
        (U[:, 1], r"$u_y(x) \,[\mathrm{mm}]$", U[:, 5], r"$\theta_z(x) \,[\degree]$"),
        (U[:, 2], r"$u_z(x) \,[\mathrm{mm}]$", U[:, 4], r"$\theta_y(x) \,[\degree]$"),
    ]

    for i, (ax_left, ax_right, (disp, disp_label, rot, rot_label)) in enumerate(zip(axes[:, 0], axes[:, 1], deformation_pairs)):
        ax_left.plot(node_positions, disp * 1000 * scale, color=color, linewidth=2, marker='o', markerfacecolor=color, markeredgecolor=color)
        ax_left.axhline(0, color='k', linestyle='--', linewidth=0.75)
        if i == 0:
            ax_left.set_title("Translation profiles", fontsize=12, fontweight='bold')
            ax_right.set_title("Rotation profiles", fontsize=12, fontweight='bold')
        ax_left.set_ylabel(disp_label, fontsize=12)
        ax_left.grid(True, linestyle='--', alpha=0.6)

        ax_right.plot(node_positions, np.degrees(rot) * scale, color=color, linewidth=2, marker='o', markerfacecolor=color, markeredgecolor=color)
        ax_right.axhline(0, color='k', linestyle='--', linewidth=0.75)
        ax_right.set_ylabel(rot_label, fontsize=12)
        ax_right.grid(True, linestyle='--', alpha=0.6)

    axes[-1, 0].set_xlabel(r"$x \,[\mathrm{m}]$", fontsize=12)
    axes[-1, 1].set_xlabel(r"$x \,[\mathrm{m}]$", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def read_deformation_file(file_path):
    """
    Reads a deformation vector from a given file.
    :param file_path: Path to the deformation results file.
    :return: Numpy array of shape (n_nodes, 6) containing deformations at each node.
    """
    try:
        with open(file_path, 'r') as file:
            data = np.loadtxt(file, comments='#')  # Ignore header lines starting with #
            data = data.reshape(-1, 6)  # Reshape into (n_nodes, 6) format
            return data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def process_deformation_files():
    """
    Cycles through deformation result files and visualizes the deformations.
    """
    base_dir = "post_processing/results"
    mesh_dir = "jobs"
    search_pattern = os.path.join(base_dir, "job_*/primary_results/job_*_static_global_U_global_*.txt")
    files = sorted(glob.glob(search_pattern))

    if not files:
        print("No deformation files found.")
        return

    for file_path in files:
        job_descriptor = os.path.basename(file_path).replace("job_", "").replace("static_global_U_global_", "").replace(".txt", "")
        job_id = os.path.basename(file_path).split('_')[1]
        mesh_path = os.path.join(mesh_dir, f"job_{job_id}", "mesh.txt")

        print(f"Processing: {file_path}")
        U = read_deformation_file(file_path)
        mesh_dictionary = parse_mesh(mesh_path)

        if U is not None and mesh_dictionary is not None and 'node_coordinates' in mesh_dictionary:
            node_positions = mesh_dictionary['node_coordinates'][:, 0]  # Extract x-coordinates
            visualize_deformation(U, node_positions, scale=1.0, title_suffix=f"job_{job_descriptor}")

# Run the script
if __name__ == "__main__":
    process_deformation_files()
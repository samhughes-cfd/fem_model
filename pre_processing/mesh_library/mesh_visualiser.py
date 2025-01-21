# pre_processing\mesh_library\mesh_visualiser.py

import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import re

# Directory containing mesh files
MESH_DIRECTORY = "pre_processing\mesh_library\meshes"

def extract_growth_factor(filename):
    """
    Extracts the growth factor (k) from the filename.
    """
    match = re.search(r'k_(\d+)', filename)
    return int(match.group(1)) if match else None

def load_mesh_from_file(filepath):
    """
    Load mesh node positions (x, y) from a given mesh file.
    """
    node_positions = []
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()
            read_nodes = False
            for line in lines:
                if line.strip().startswith("[node_ids]"):
                    read_nodes = True
                    continue
                if read_nodes and line.strip() and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            x, y = float(parts[1]), float(parts[2])
                            node_positions.append((x, y))
                        except ValueError:
                            logging.warning(f"Skipping invalid line: {line.strip()}")

        if not node_positions:
            logging.error(f"No node positions found in the file: {filepath}")
            return None, None

    except FileNotFoundError:
        logging.error(f"Mesh file '{filepath}' not found.")
        return None, None

    num_elements = len(node_positions) - 1
    return np.array(node_positions), num_elements

def visualize_meshes():
    """
    Visualize multiple mesh distributions in stacked subplots.
    """
    if not os.path.exists(MESH_DIRECTORY):
        logging.error(f"Mesh directory '{MESH_DIRECTORY}' does not exist.")
        return

    mesh_files = sorted([f for f in os.listdir(MESH_DIRECTORY) if f.endswith(".txt")])
    if not mesh_files:
        logging.error("No mesh files found in the directory.")
        return

    fig, axes = plt.subplots(len(mesh_files), 1, figsize=(10, 2 * len(mesh_files)), sharex=True)
    if len(mesh_files) == 1:
        axes = [axes]

    for ax, mesh_file in zip(axes, mesh_files):
        mesh_path = os.path.join(MESH_DIRECTORY, mesh_file)
        node_positions, num_elements = load_mesh_from_file(mesh_path)

        if node_positions is None:
            continue

        x_vals, y_vals = node_positions[:, 0], node_positions[:, 1]
        ax.plot(x_vals, y_vals, 'o-', markersize=4, label=mesh_file)
        ax.set_ylabel("Y Position [m]")
        ax.legend()
        ax.grid(True)

    plt.xlabel("X Position [m]")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    visualize_meshes()
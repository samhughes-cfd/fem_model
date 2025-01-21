# pre_processing\mesh_library\schemes\basic_mesh_generator.py

import numpy as np
import logging
import os
from datetime import datetime

# Configuration Parameters (User-defined)
L = 8.0  # Beam length [m]
growth_factor = 2  # Exponential distribution parameter (set to 0 for uniform mesh)
max_num_nodes = 101  # Maximum number of nodes allowed if using growth factor
num_uniform_nodes = 1000  # Exact number of nodes if using a uniform mesh (growth_factor = 0)

def generate_mesh(growth_factor, max_num_nodes, num_uniform_nodes, tolerance=1e-6):
    """
    Generate mesh nodes and elements for a cantilever beam.
    """
    if growth_factor == 0:
        num_nodes = num_uniform_nodes
    else:
        num_nodes = min(max_num_nodes, 1000)  # Ensuring it's within reasonable bounds

    # Generate node positions
    i = np.linspace(0, 1, num_nodes)

    if growth_factor == 0:
        # Uniform spacing
        node_positions = i * L
    else:
        # Exponential spacing
        normalized_positions = (np.exp(growth_factor * i) - 1) / (np.exp(growth_factor) - 1)
        node_positions = (1 - normalized_positions) * L  # Reverse for tip clustering

    # Remove duplicate nodes within tolerance
    node_positions = np.unique(np.round(node_positions / tolerance) * tolerance)

    # Define elements
    elements = [(idx + 1, idx + 2) for idx in range(len(node_positions) - 1)]

    logging.info(f"Mesh generation successful.")
    logging.info(f"Growth Factor: {growth_factor}")
    logging.info(f"Total Nodes Generated: {len(node_positions)}")
    logging.info(f"Total Elements Generated: {len(elements)}")

    return node_positions, elements

def save_mesh_to_file(node_positions, elements, save_directory='mesh_library/meshes'):
    """
    Save the mesh nodes and elements to a text file.
    """
    os.makedirs(save_directory, exist_ok=True)

    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mesh_{timestamp}.txt"
    filepath = os.path.join(save_directory, filename)

    # Compute spacing adjustment based on manual alignment
    spacing_adjustment = 1  # Adjust this value based on observed misalignment

    with open(filepath, 'w') as f:
        f.write("[element_types]\n")
        f.write("EulerBernoulliBeamElement\n\n")
        f.write("[node_ids]   [x]          [y]        [z]        [connectivity]\n")

        for idx, x_pos in enumerate(node_positions):
            node_id = idx + 1
            connectivity = "-" if idx == len(node_positions) - 1 else f"({node_id}, {node_id + 1})"
            f.write(f"{node_id:<{12+spacing_adjustment}}{x_pos:<{12+spacing_adjustment}.6f}{0.0:<{10+spacing_adjustment}.1f}{0.0:<{10+spacing_adjustment}.1f}{connectivity}\n")

    logging.info(f"Mesh file saved to '{filepath}'.")

def main():
    """
    Main function to generate and save the mesh.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        nodes, elems = generate_mesh(growth_factor, max_num_nodes, num_uniform_nodes)
        print(f"Number of nodes: {len(nodes)}")
        print(f"Number of elements: {len(elems)}")
        save_mesh_to_file(nodes, elems)
    except Exception as e:
        logging.error(f"An error occurred during mesh generation: {e}")

if __name__ == "__main__":
    main()
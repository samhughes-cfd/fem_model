# pre_processing\mesh_library\schemes\mesh_generator.py

import numpy as np
import logging
import os
from datetime import datetime

# Configuration Parameters (User-defined)
L = 2.0                    # Beam length [m]
growth_factor = 0         # Exponential distribution parameter (0 for uniform)
num_uniform_nodes = 101   # Number of nodes if using uniform spacing
max_num_nodes = 101       # Max nodes if using exponential spacing


def generate_mesh(growth_factor, max_num_nodes, num_uniform_nodes):
    """
    Generate mesh nodes and elements for a cantilever beam.
    Returns node_positions and element connectivity tuples (start, end).
    """
    if growth_factor == 0:
        # Uniform mesh
        node_positions = np.linspace(0, L, num_uniform_nodes)
    else:
        # Exponential mesh using growth factor, with exactly max_num_nodes
        i = np.linspace(0, 1, max_num_nodes)
        normalized_positions = (np.exp(growth_factor * i) - 1) / (np.exp(growth_factor) - 1)
        node_positions = (1 - normalized_positions) * L  # Tip clustering

    # Guarantee no trimming: don't apply rounding or unique()
    elements = [(idx, idx + 1) for idx in range(len(node_positions) - 1)]

    logging.info(f"Mesh generation successful.")
    logging.info(f"Growth Factor: {growth_factor}")
    logging.info(f"Total Nodes Generated: {len(node_positions)}")
    logging.info(f"Total Elements Generated: {len(elements)}")

    return node_positions, elements

def save_grid_file(node_positions, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    filepath = os.path.join(save_directory, 'grid.txt')

    with open(filepath, 'w') as f:
        f.write("[Grid]\n")
        f.write(f"{'[node_id]':<12}{'[x]':<12}{'[y]':<10}{'[z]':<10}\n")
        for i, x in enumerate(node_positions):
            f.write(f"{i:<12d}{x:<12.6f}{0.0:<10.1f}{0.0:<10.1f}\n")

    logging.info(f"grid.txt saved to '{filepath}'.")

def save_element_file(elements, save_directory, element_type="EulerBernoulliBeamElement3D"):
    filepath = os.path.join(save_directory, 'element.txt')
    os.makedirs(save_directory, exist_ok=True)

    axial_order = 3
    bending_y_order = 3
    bending_z_order = 3
    shear_y_order = 0
    shear_z_order = 0
    torsion_order = 3
    load_order = 2

    with open(filepath, 'w') as f:
        f.write("[Element]\n")
        f.write(f"{'[element_id]':<14}{'[node1]':<9}{'[node2]':<9}{'[element_type]':<30}"
                f"{'[axial_order]':<15}{'[bending_y_order]':<18}{'[bending_z_order]':<18}"
                f"{'[shear_y_order]':<17}{'[shear_z_order]':<16}{'[torsion_order]':<15}{'[load_order]'}\n")

        for i, (n1, n2) in enumerate(elements):
            f.write(f"{i:<14d}{n1:<9d}{n2:<9d}{element_type:<30}"
                    f"{axial_order:<15d}{bending_y_order:<18d}{bending_z_order:<18d}"
                    f"{shear_y_order:<17d}{shear_z_order:<16d}{torsion_order:<15d}{load_order}\n")

    logging.info(f"element.txt saved to '{filepath}'.")

def save_material_file(elements, save_directory):
    filepath = os.path.join(save_directory, 'material.txt')
    os.makedirs(save_directory, exist_ok=True)

    E = 2e11
    G = 7.6923e10
    nu = 0.3
    rho = 7850

    with open(filepath, 'w') as f:
        f.write("[Material]\n")
        f.write(f"{'[element_id]':<14}{'[E]':<12}{'[G]':<13}{'[nu]':<7}{'[rho]':<10}\n")
        for i in range(len(elements)):
            f.write(f"{i:<14d}{E:<12.1e}{G:<13.4e}{nu:<7.1f}{rho:<10d}\n")

    logging.info(f"material.txt saved to '{filepath}'.")

def save_section_file(elements, save_directory):
    filepath = os.path.join(save_directory, 'section.txt')
    os.makedirs(save_directory, exist_ok=True)

    A = 0.02
    I_x = 0.0
    I_y = 6.6667e-5
    I_z = 1.6667e-5
    J_t = 1.707e-5

    with open(filepath, 'w') as f:
        f.write("[Section]\n")
        f.write(f"{'[element_id]':<14}{'[A]':<10}{'[I_x]':<10}{'[I_y]':<13}{'[I_z]':<13}{'[J_t]'}\n")
        for i in range(len(elements)):
            f.write(f"{i:<14d}{A:<10.5f}{I_x:<10.1f}{I_y:<13.5e}{I_z:<13.5e}{J_t:.5e}\n")

    logging.info(f"section.txt saved to '{filepath}'.")

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        # Generate the mesh
        nodes, elems = generate_mesh(growth_factor, max_num_nodes, num_uniform_nodes)
        print(f"Number of nodes: {len(nodes)}")
        print(f"Number of elements: {len(elems)}")

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = r'pre_processing\mesh_library\meshes'
        save_dir = os.path.join(base_dir, f"mesh_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        # Save all mesh component files to the timestamped directory
        save_grid_file(nodes, save_dir)
        save_element_file(elems, save_dir)
        save_material_file(elems, save_dir)
        save_section_file(elems, save_dir)

        logging.info(f"All mesh files saved to '{save_dir}'.")

    except Exception as e:
        logging.error(f"An error occurred during mesh generation: {e}")

if __name__ == "__main__":
    main()
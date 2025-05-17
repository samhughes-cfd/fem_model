# pre_processing\mesh_library\schemes\basic_mesh_generator.py

import numpy as np
import logging
import os
from datetime import datetime

# Configuration Parameters (User-defined)
L = 2.0                 # Beam length [m]
growth_factor = 0       # Exponential distribution parameter (set to 0 for uniform mesh)
num_uniform_nodes = 101   # Number of nodes for uniform mesh
max_num_nodes = 100     # Maximum number of nodes allowed if using growth factor

def generate_mesh(growth_factor, max_num_nodes, num_uniform_nodes, tolerance=1e-6):
    """
    Generate mesh nodes and elements for a cantilever beam.

    Parameters:
        growth_factor (float): Parameter for exponential node distribution. Set to 0 for uniform distribution.
        max_num_nodes (int): Maximum number of nodes allowed if using growth factor.
        num_uniform_nodes (int): Exact number of nodes if using a uniform mesh (growth_factor = 0).
        tolerance (float): Tolerance for removing duplicate nodes.

    Returns:
        node_positions (np.ndarray): Array of node positions along the beam.
        elements (list of tuples): List of element connectivities as (start_node, end_node).
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

def save_mesh_to_file(node_positions, elements, element_type='EulerBernoulliBeamElement3D', save_directory=r'pre_processing\mesh_library\meshes'):
    """
    Save the mesh nodes and elements to a text file with an additional element type column.

    Parameters:
        node_positions (np.ndarray): Array of node positions along the beam.
        elements (list of tuples): List of element connectivities as (start_node, end_node).
        element_type (str): Type of the elements to be listed. Default is 'EulerBernoulliBeamElement'.
        save_directory (str): Directory where the mesh file will be saved.
    """
    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mesh_{timestamp}.txt"
    filepath = os.path.join(save_directory, filename)

    with open(filepath, 'w') as f:
        # Write the [mesh] section header
        f.write("[Mesh]\n")
        
        # Define column headers with fixed-width formatting
        headers = [
            "[node_ids]", "[x]", "[y]", "[z]", "[connectivity]", "[element_type]"
        ]
        
        # Define corresponding field widths for alignment
        # Adjust field widths as needed to accommodate your data
        field_widths = [15, 15, 12, 12, 20, 25]
        
        # Create the header line using fixed-width formatting
        header_line = ""
        for header, width in zip(headers, field_widths):
            header_line += f"{header:<{width}}"
        header_line += "\n"
        
        f.write(header_line)
        
        # Iterate over nodes to write their data
        for idx, x_pos in enumerate(node_positions):
            node_id = idx + 1
            if idx < len(node_positions) - 1:
                connectivity = f"({node_id},{node_id + 1})"
                current_element_type = element_type
            else:
                connectivity = "-"
                current_element_type = "-"
            # Create the data line using fixed-width formatting
            data_line = (
                f"{node_id:<{field_widths[0]}}"
                f"{x_pos:<{field_widths[1]}.6f}"
                f"{0.0:<{field_widths[2]}.1f}"
                f"{0.0:<{field_widths[3]}.1f}"
                f"{connectivity:<{field_widths[4]}}"
                f"{current_element_type:<{field_widths[5]}}"
                "\n"
            )
            f.write(data_line)

    logging.info(f"Mesh file saved to '{filepath}'.")

def main():
    """
    Main function to generate and save the mesh.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        # Generate mesh data
        nodes, elems = generate_mesh(growth_factor, max_num_nodes, num_uniform_nodes)
        
        # Display the number of nodes and elements
        print(f"Number of nodes: {len(nodes)}")
        print(f"Number of elements: {len(elems)}")
        
        # Save the mesh to file
        save_mesh_to_file(nodes, elems)
        
    except Exception as e:
        logging.error(f"An error occurred during mesh generation: {e}")

if __name__ == "__main__":
    main()
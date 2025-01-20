# pre_processing\parsing\mesh_parser.py

import ast
import logging
import numpy as np


def parse_mesh(mesh_file_path):
    """
    Parses a new-form mesh file and computes element lengths using node coordinates.

    Args:
        mesh_file_path (str): Path to the mesh file.

    Returns:
        dict: {
            'element_types': List of element types,
            'node_ids': List of node IDs,
            'node_positions': NumPy array of node positions (shape: [num_nodes, 3]),
            'connectivity': List of element connectivity tuples,
            'element_lengths': Dictionary {element_id: length}
        }
    """
    element_types = []
    node_ids = []
    node_positions = []
    connectivity_list = []

    current_section = None

    with open(mesh_file_path, 'r') as f:
        lines = f.readlines()

    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Remove inline comments
        if "#" in line:
            line = line.split("#", 1)[0].strip()

        # Identify sections
        lower_line = line.lower()
        if lower_line.startswith("[element_types]"):
            current_section = "element_types"
            continue
        elif lower_line.startswith("[node_ids]"):
            current_section = "nodes"
            continue

        # Parse content based on the current section
        if current_section == "element_types":
            element_types.append(line)

        elif current_section == "nodes":
            # Skip header row
            if "[" in line and "]" in line:
                logging.info(f"Line {line_number}: Skipping header line '{raw_line}'")
                continue

            # Expecting: node_id, x, y, z, connectivity
            columns = line.split(maxsplit=4)
            if len(columns) < 5:
                logging.warning(f"Line {line_number}: Incomplete node data: '{raw_line}'")
                continue

            try:
                node_id = int(columns[0])
                x, y, z = map(float, columns[1:4])  # Read full 3D coordinates
                conn_str = columns[4].strip()  # e.g., "(1, 2)" or "-"

                node_ids.append(node_id)
                node_positions.append((x, y, z))  # Store as tuple

                if conn_str != "-":
                    try:
                        c_tuple = ast.literal_eval(conn_str)
                        if isinstance(c_tuple, tuple) and len(c_tuple) == 2 and all(isinstance(i, int) for i in c_tuple):
                            connectivity_list.append(c_tuple)
                        else:
                            logging.warning(f"Line {line_number}: Invalid connectivity tuple: {conn_str}")
                    except (ValueError, SyntaxError) as e:
                        logging.warning(f"Line {line_number}: Error parsing connectivity '{conn_str}': {e}")
                else:
                    # No connectivity for this node
                    pass

            except ValueError as e:
                logging.warning(f"Line {line_number}: Invalid node or position: '{raw_line}' ({e})")
                continue

    # Convert node positions to a NumPy array
    node_positions = np.array(node_positions)

    # Compute element lengths using Euclidean distance
    element_lengths = compute_element_lengths(connectivity_list, node_positions, node_ids)

    return {
        'element_types': element_types,
        'node_ids': node_ids,
        'node_positions': node_positions,  # Shape: (num_nodes, 3)
        'connectivity': connectivity_list,
        'element_lengths': element_lengths
    }


def compute_element_lengths(connectivity_list, node_positions, node_ids):
    """
    Computes the length of each element based on 3D node positions.

    Args:
        connectivity_list (list of tuples): List of element connectivity (node1, node2).
        node_positions (np.array): 3D positions of nodes (shape: [num_nodes, 3]).
        node_ids (list): List of node IDs.

    Returns:
        dict: {element_id: length} mapping element indices to their computed lengths.
    """
    element_lengths = {}

    for element_id, (node1, node2) in enumerate(connectivity_list):
        try:
            # Find the actual index in node_positions based on node_id mapping
            index1 = node_ids.index(node1)
            index2 = node_ids.index(node2)

            # Compute 3D Euclidean distance
            length = np.linalg.norm(node_positions[index2] - node_positions[index1])
            element_lengths[element_id] = length
        except ValueError:
            logging.error(f"Element {element_id} references undefined nodes {node1}, {node2}")
            continue

    return element_lengths
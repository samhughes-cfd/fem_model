# re_processing\parsing\mesh_parser.py

import ast
import logging
import numpy as np
from .geometry_parser import parse_geometry  # Importing function to get L

def parse_mesh(mesh_file_path, geometry_file_path):
    """
    Parses a bracket-based mesh file and computes element lengths using normalized positions.

    Args:
        mesh_file_path (str): Path to the mesh file.
        geometry_file_path (str): Path to the geometry file.

    Returns:
        dict: {
            'element_types': List of element types,
            'node_ids': List of node IDs,
            'node_positions': NumPy array of *scaled* node positions,
            'connectivity': List of element connectivity tuples,
            'element_lengths': Dictionary {element_id: length}
        }
    """
    # Retrieve beam length from geometry.txt
    geometry_data = parse_geometry(geometry_file_path)
    beam_length = geometry_data.get("L", None)

    if beam_length is None:
        raise ValueError("Beam length `L` not found in geometry file.")

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

        # Remove inline comment
        if '#' in line:
            line = line.split('#', 1)[0].strip()

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
            # Each line is an element type
            element_types.append(line)

        elif current_section == "nodes":
            # If this is the header line (with bracketed columns), skip
            if "[" in line and "]" in line:
                logging.info(f"Line {line_number}: Skipping node header line '{raw_line}'")
                continue

            # Split into exactly 3 parts: node_id, norm_position, connectivity
            columns = line.split(maxsplit=2)
            if len(columns) < 3:
                logging.warning(f"Line {line_number}: Incomplete node data: '{raw_line}'")
                continue

            try:
                node_id = int(columns[0])
                norm_position = float(columns[1])  # Normalized in range [0,1]
                conn_str = columns[2].strip()  # e.g. "(1, 2)" or "-"

                node_ids.append(node_id)
                node_positions.append(norm_position * beam_length)  # Scale to actual length

                if conn_str != "-":
                    try:
                        c_tuple = ast.literal_eval(conn_str)
                        if isinstance(c_tuple, tuple):
                            connectivity_list.append(c_tuple)
                        else:
                            logging.warning(
                                f"Line {line_number}: Invalid connectivity (not a tuple): {conn_str}"
                            )
                    except (ValueError, SyntaxError) as e:
                        logging.warning(
                            f"Line {line_number}: Error parsing connectivity '{conn_str}': {e}"
                        )
                else:
                    # no connectivity
                    pass

            except ValueError as e:
                logging.warning(f"Line {line_number}: Invalid node or position: '{raw_line}' ({e})")
                continue

    # Convert node positions to a NumPy array
    node_positions = np.array(node_positions)

    # Compute element lengths based on connectivity
    element_lengths = compute_element_lengths(connectivity_list, node_positions)

    return {
        'element_types': element_types,
        'node_ids': node_ids,
        'node_positions': node_positions,  # Now scaled to actual beam length
        'connectivity': connectivity_list,
        'element_lengths': element_lengths
    }


def compute_element_lengths(connectivity_list, node_positions):
    """
    Computes the length of each element based on scaled node positions.

    Args:
        connectivity_list (list of tuples): List of element connectivity (node1, node2).
        node_positions (np.array): *Scaled* positions of nodes along the beam.

    Returns:
        dict: {element_id: length} mapping element indices to their computed lengths.
    """
    element_lengths = {}

    for element_id, (node1, node2) in enumerate(connectivity_list):
        try:
            # Compute length as the absolute difference between node positions
            length = abs(node_positions[node2 - 1] - node_positions[node1 - 1])
            element_lengths[element_id] = length
        except IndexError:
            logging.error(f"Element {element_id} references undefined nodes {node1}, {node2}")
            continue

    return element_lengths
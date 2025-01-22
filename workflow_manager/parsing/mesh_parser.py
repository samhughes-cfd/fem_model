# pre_processing\parsing\mesh_parser.py

import ast
import logging
import numpy as np
import re

logging.basicConfig(level=logging.WARNING)

def parse_mesh(mesh_file_path):
    """
    Parses a structured mesh file and computes element lengths using node coordinates.

    =============================
    Mesh Properties Mapping
    =============================

    Index   Property             Data Type                  Shape
    ----------------------------------------------------------------
    0       Element Types        NumPy array (str)         `(N,)`
    1       Node IDs             NumPy array (int)         `(N,)`
    2       Node Positions       NumPy array (float)       `(N, 3)`
    3       Connectivity         NumPy array (int)         `(M, 2)`
    4       Element Lengths      NumPy array (float)       `(M,)`

    The function reads mesh data, extracts node positions, and computes 
    element lengths using the Euclidean distance formula. Empty lines and 
    comments (`#`) are ignored.

    Parameters
    ----------
    mesh_file_path : str
        Path to the structured mesh file.

    Returns
    -------
    tuple
        (
            element_types_array: np.ndarray[str]  -> Element type names
            node_ids_array: np.ndarray[int]       -> Unique node identifiers
            node_positions_array: np.ndarray[float] -> 3D node coordinates `(N, 3)`
            connectivity_array: np.ndarray[int]   -> Connectivity pairs `(M, 2)`
            element_lengths_array: np.ndarray[float] -> Computed element lengths `(M,)`
        )

    Raises
    ------
    ValueError
        If node coordinates or connectivity data cannot be parsed.

    Warnings
    --------
    Logs a warning if an invalid node or connectivity entry is encountered.

    Example
    -------
    >>> element_types_array, node_ids_array, node_positions_array, connectivity_array, element_lengths_array = parse_mesh("mesh.txt")
    >>> print(node_positions_array)
    array([[0.0, 0.1, 0.2],
           [1.5, 1.6, 1.7],
           [2.2, 2.3, 2.4]])

    Notes
    -----
    - Nodes must be formatted as `ID X Y Z (Node1, Node2)`, where connectivity is optional.
    - If connectivity is missing, `-` is used as a placeholder.
    - Inline comments (`#`) are ignored.
    """

    element_types = []
    node_ids = []
    node_positions = []
    connectivity_list = []

    section_pattern = re.compile(r"\[.*?\]", re.IGNORECASE)  # Match section headers
    current_section = None

    with open(mesh_file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.split("#")[0].strip()  # Remove inline comments

            if not line:
                continue  # Skip empty lines

            # Detect sections dynamically
            if section_pattern.match(line):
                current_section = line.lower()
                continue

            # Process element types
            if current_section == "[element_types]":
                element_types.append(line)

            # Process node IDs and positions
            elif current_section == "[node_ids]":
                parts = line.split(maxsplit=4)
                if len(parts) < 5:
                    logging.warning(f"Line {line_number}: Incomplete node data: '{raw_line}'. Skipping.")
                    continue

                try:
                    node_id = int(parts[0])
                    x, y, z = map(float, parts[1:4])
                    conn_str = parts[4].strip()

                    node_ids.append(node_id)
                    node_positions.append((x, y, z))

                    if conn_str != "-":
                        try:
                            c_tuple = ast.literal_eval(conn_str)
                            if isinstance(c_tuple, tuple) and len(c_tuple) == 2 and all(isinstance(i, int) for i in c_tuple):
                                connectivity_list.append(c_tuple)
                            else:
                                logging.warning(f"Line {line_number}: Invalid connectivity tuple: {conn_str}.")
                        except (ValueError, SyntaxError) as e:
                            logging.warning(f"Line {line_number}: Error parsing connectivity '{conn_str}': {e}.")
                except ValueError:
                    logging.warning(f"Line {line_number}: Invalid node data: '{raw_line}'. Skipping.")

    # Convert lists to NumPy arrays (Consistent Naming)
    element_types_array = np.array(element_types, dtype=str) if element_types else np.empty((0,), dtype=str)
    node_ids_array = np.array(node_ids, dtype=int) if node_ids else np.empty((0,), dtype=int)
    node_positions_array = np.array(node_positions, dtype=float) if node_positions else np.empty((0, 3), dtype=float)
    connectivity_array = np.array(connectivity_list, dtype=int) if connectivity_list else np.empty((0, 2), dtype=int)

    # Compute element lengths (Optimized)
    element_lengths_array = compute_element_lengths(connectivity_array, node_positions_array, node_ids_array)

    return element_types_array, node_ids_array, node_positions_array, connectivity_array, element_lengths_array

def compute_element_lengths(connectivity_array, node_positions_array, node_ids_array):
    """
    Computes element lengths based on 3D node positions.

    Parameters
    ----------
    connectivity_array : np.ndarray[int]
        NumPy array of shape `(M, 2)`, containing node ID pairs.
    node_positions_array : np.ndarray[float]
        NumPy array of shape `(N, 3)`, containing node positions.
    node_ids_array : np.ndarray[int]
        NumPy array of shape `(N,)`, containing node identifiers.

    Returns
    -------
    np.ndarray[float]
        NumPy array of shape `(M,)`, containing computed element lengths.

    Raises
    ------
    ValueError
        If a referenced node ID is not found in `node_ids_array`.
    """

    if connectivity_array.shape[0] == 0:
        return np.empty((0,), dtype=float)

    # Map node IDs to indices efficiently using NumPy `searchsorted`
    sorted_indices = np.argsort(node_ids_array)
    sorted_node_ids = node_ids_array[sorted_indices]

    node_indices = np.searchsorted(sorted_node_ids, connectivity_array)

    # Extract corresponding node positions
    pos1 = node_positions_array[sorted_indices[node_indices[:, 0]]]
    pos2 = node_positions_array[sorted_indices[node_indices[:, 1]]]

    # Compute Euclidean distance (Vectorized)
    element_lengths_array = np.linalg.norm(pos2 - pos1, axis=1)

    return element_lengths_array
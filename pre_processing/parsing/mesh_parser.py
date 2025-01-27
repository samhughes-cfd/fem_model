# pre_processing\parsing\mesh_parser.py

import ast
import logging
import numpy as np
import re
import os

logging.basicConfig(level=logging.WARNING)
#logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)

def parse_mesh(mesh_file_path):
    """
    Parses a structured mesh file and computes element lengths using node coordinates.

    =============================
    Mesh Properties Mapping
    =============================

    Index   Property             Key in Dictionary         Data Type             Shape     Units  
    ------------------------------------------------------------------------------------------------
    0       Element Types        `element_types`          `np.ndarray[str]`      (N,)       -      
    1       Node IDs             `node_ids`               `np.ndarray[int]`      (N,)       -      
    2       Node Positions       `node_coordinates`       `np.ndarray[float]`    (N, 3)     [m] 
    3       Connectivity         `connectivity`           `np.ndarray[int]`      (M, 2)     -      
    4       Element Lengths      `element_lengths`        `np.ndarray[float]`    (M,)       [m] 
    5       Element IDs          `element_ids`            `np.ndarray[int]`      (M,)       -      
        

    The function reads mesh data, extracts node positions, and computes 
    element lengths using the Euclidean distance formula. Empty lines and 
    comments (#) are ignored.

    Parameters
    ----------
    mesh_file_path : str
        Path to the structured mesh file.

    Returns
    -------
    dict
        Dictionary "mesh_dictionary" with the following keys:
            - 'element_types': np.ndarray[str]
            - 'node_ids': np.ndarray[int]
            - 'node_coordinates': np.ndarray[float]
            - 'connectivity': np.ndarray[int]
            - 'element_lengths': np.ndarray[float]
            - 'element_ids': np.ndarray[int]  # Included in Returns

    Raises
    ------
    FileNotFoundError
        If the mesh file does not exist.
    ValueError
        If node coordinates or connectivity data cannot be parsed.

    Warnings
    --------
    Logs a warning if an invalid node or connectivity entry is encountered.

    Example
    -------
    >>> mesh_dictionary = parse_mesh("mesh.txt")
    >>> print(mesh_dictionary['element_ids'])
    array([1, 2, 3, ...])

    Notes
    -----
    - Nodes must be formatted as `ID X Y Z (Node1, Node2)`, where connectivity is optional.
    - If connectivity is missing, `-` is used as a placeholder.
    - Inline comments (#) are ignored.
    """

    # 1. Ensure the file exists
    if not os.path.exists(mesh_file_path):
        logging.error(f"[Mesh] File not found: {mesh_file_path}")
        raise FileNotFoundError(f"{mesh_file_path} not found")

    # 2. Prepare storage and regex
    element_types = []
    node_ids = []
    node_coordinates = []
    connectivity_list = []

    # Regex to strictly match lines with only [Mesh] or [Element_Types]
    header_mesh_pattern = re.compile(r"^\[mesh\]$", re.IGNORECASE)
    header_element_types_pattern = re.compile(r"^\[element_types\]$", re.IGNORECASE)

    current_section = None
    found_mesh_section = False
    found_element_types_section = False

    # 3. Read file line-by-line
    with open(mesh_file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            # Strip out inline comments (#...) and trailing/leading whitespace
            line = raw_line.split("#")[0].strip()

            logging.debug(f"[Mesh] Processing line {line_number}: '{raw_line.strip()}'")

            # Skip completely empty lines
            if not line:
                logging.debug(f"[Mesh] Line {line_number} is empty. Skipping.")
                continue

            # --- Section detection -----------------------------------------
            if header_element_types_pattern.match(line):
                logging.info(f"[Mesh] Found [Element_Types] at line {line_number}.")
                current_section = "element_types"
                found_element_types_section = True
                continue

            if header_mesh_pattern.match(line):
                logging.info(f"[Mesh] Found [Mesh] at line {line_number}. Parsing mesh data.")
                current_section = "mesh"
                found_mesh_section = True
                continue

            # --- Element_Types section -------------------------------------
            if current_section == "element_types":
                element_types.append(line)
                logging.debug(f"[Mesh] Parsed element type: {line}")
                continue

            # --- Mesh section ----------------------------------------------
            if current_section == "mesh":
                # If line includes 'node_ids', 'x', 'y', 'z' (a column header), skip it
                headers = ["node_ids", "x", "y", "z"]
                if all(header in line.lower() for header in headers):
                    logging.debug(f"[Mesh] Skipping column header at line {line_number}.")
                    continue

                # Expect exactly 5 tokens for a valid node line:
                # node_id, x, y, z, connectivity
                parts = line.split(maxsplit=4)
                if len(parts) < 5:
                    logging.warning(f"[Mesh] Line {line_number}: Incomplete node data. Skipping.")
                    continue

                try:
                    node_id = int(parts[0])
                    x, y, z = map(float, parts[1:4])
                    conn_str = parts[4].strip()

                    node_ids.append(node_id)
                    node_coordinates.append((x, y, z))
                    logging.debug(f"[Mesh] Parsed node {node_id}: ({x}, {y}, {z})")

                    # If connectivity is not '-', parse the tuple
                    if conn_str != "-":
                        try:
                            c_tuple = ast.literal_eval(conn_str)
                            if (isinstance(c_tuple, tuple) and 
                                len(c_tuple) == 2 and 
                                all(isinstance(i, int) for i in c_tuple)):
                                connectivity_list.append(c_tuple)
                                logging.debug(f"[Mesh] Parsed connectivity: {c_tuple}")
                            else:
                                logging.warning(f"[Mesh] Line {line_number}: Invalid connectivity format: {conn_str}")
                        except (ValueError, SyntaxError) as e:
                            logging.warning(f"[Mesh] Line {line_number}: Connectivity parse error '{conn_str}': {e}")
                except ValueError:
                    logging.warning(f"[Mesh] Line {line_number}: Invalid node data. Skipping.")
                continue

            # If we're here, we're outside [Element_Types] or [Mesh]
            logging.warning(f"[Mesh] Line {line_number} ignored: Outside relevant sections.")
        # end for line_number, raw_line in ...

    # 4. Check if we found the key sections
    if not found_mesh_section:
        logging.warning(f"[Mesh] No [Mesh] section found in '{mesh_file_path}'. Returning empty arrays.")
        mesh_dictionary = {
            'element_types': np.empty((0,), dtype=str),
            'node_ids': np.empty((0,), dtype=int),
            'node_coordinates': np.empty((0, 3), dtype=float),
            'connectivity': np.empty((0, 2), dtype=int),
            'element_lengths': np.empty((0,), dtype=float),
            'element_ids': np.empty((0,), dtype=int)  # Added Empty Array
        }
        return mesh_dictionary

    if not found_element_types_section:
        logging.warning(f"[Mesh] No [Element_Types] section found in '{mesh_file_path}'. Returning empty element types array.")
        element_types_array = np.empty((0,), dtype=str)
    else:
        element_types_array = np.array(element_types, dtype=str)

    if not node_ids:
        logging.error(f"[Mesh] No valid node data found in '{mesh_file_path}'. Returning empty arrays.")
        mesh_dictionary = {
            'element_types': element_types_array,
            'node_ids': np.empty((0,), dtype=int),
            'node_coordinates': np.empty((0, 3), dtype=float),
            'connectivity': np.empty((0, 2), dtype=int),
            'element_lengths': np.empty((0,), dtype=float),
            'element_ids': np.empty((0,), dtype=int)  # Added Empty Array
        }
        return mesh_dictionary

    # 5. Convert data to NumPy arrays
    node_ids_array = np.array(node_ids, dtype=int)
    node_coordinates_array = np.array(node_coordinates, dtype=float)
    connectivity_array = np.array(connectivity_list, dtype=int) if connectivity_list else np.empty((0, 2), dtype=int)

    # 6. Compute element lengths
    element_lengths_array = compute_element_lengths(connectivity_array, node_coordinates_array, node_ids_array)

    # 7. Generate element IDs starting from 1
    element_ids_array = np.arange(1, connectivity_array.shape[0] + 1, dtype=int)  # Updated Line

    # 8. Construct the mesh_dictionary
    mesh_dictionary = {
        'element_types': element_types_array,
        'node_ids': node_ids_array,
        'node_coordinates': node_coordinates_array,
        'connectivity': connectivity_array,
        'element_lengths': element_lengths_array,
        'element_ids': element_ids_array  # Added Key-Value Pair
    }

    # Log final stats
    logging.info(
        f"[Mesh] Parsed {len(element_types_array)} element type(s), "
        f"{len(node_ids_array)} nodes, and {len(connectivity_list)} elements from '{mesh_file_path}'."
    )
    logging.debug(
        f"[Mesh] Final parsed data:\n"
        f"  Element Types:\n{element_types_array}\n"
        f"  Nodes:\n{node_coordinates_array}\n"
        f"  Connectivity:\n{connectivity_array}\n"
        f"  Element Lengths:\n{element_lengths_array}\n"
        f"  Element IDs:\n{element_ids_array}"  # Added Debug Information
    )

    return mesh_dictionary

def compute_element_lengths(connectivity_array, node_coordinates_array, node_ids_array):
    """
    Computes element lengths based on 3D node positions.

    Parameters
    ----------
    connectivity_array : np.ndarray[int]
        NumPy array of shape (M, 2), containing node ID pairs.
    node_coordinates_array : np.ndarray[float]
        NumPy array of shape (N, 3), containing node positions.
    node_ids_array : np.ndarray[int]
        NumPy array of shape (N,), containing node identifiers.

    Returns
    -------
    np.ndarray[float]
        NumPy array of shape (M,), containing computed element lengths.

    Raises
    ------
    ValueError
        If a referenced node ID is not found in node_ids_array.
    """

    if connectivity_array.shape[0] == 0:
        logging.debug("[Mesh] No connectivity data found. Returning empty length array.")
        return np.empty((0,), dtype=float)

    # Sort node IDs so we can index them
    sorted_indices = np.argsort(node_ids_array)
    sorted_node_ids = node_ids_array[sorted_indices]

    # Map the connectivity node IDs to their sorted indices
    node_indices = np.searchsorted(sorted_node_ids, connectivity_array)

    # Check for invalid node indices
    invalid_mask = (node_indices[:, 0] >= len(sorted_node_ids)) | (node_indices[:, 1] >= len(sorted_node_ids))
    if np.any(invalid_mask):
        invalid_pairs = connectivity_array[invalid_mask]
        raise ValueError(f"Referenced node IDs not found: {invalid_pairs}")

    # Retrieve the node coordinates
    coord1 = node_coordinates_array[sorted_indices[node_indices[:, 0]]]
    coord2 = node_coordinates_array[sorted_indices[node_indices[:, 1]]]

    # Compute Euclidean distances
    element_lengths_array = np.linalg.norm(coord2 - coord1, axis=1)
    logging.debug(f"[Mesh] Computed element lengths: {element_lengths_array}")

    return element_lengths_array

# Standalone execution for quick testing
if __name__ == "__main__":
    test_file = r"jobs\job_0001\mesh.txt"  # Change to your actual path
    if not os.path.exists(test_file):
        logging.error(f"Test file '{test_file}' not found. Make sure it exists before running.")
    else:
        try:
            mesh_dictionary = parse_mesh(test_file)
            print("\n-------------Parsed Mesh Data-------------")
            if mesh_dictionary['node_ids'].size > 0:
                print(f"-----Element Types-----\n{mesh_dictionary['element_types']}")
                print(f"-----Node IDs-----\n{mesh_dictionary['node_ids']}")
                print(f"-----Node Coordinates-----\n{mesh_dictionary['node_coordinates']}")
                print(f"-----Connectivity-----\n{mesh_dictionary['connectivity']}")
                print(f"-----Element Lengths-----\n{mesh_dictionary['element_lengths']}")
                print(f"-----Element IDs-----\n{mesh_dictionary['element_ids']}")  # Added Print Statement
            else:
                print("No data parsed.")
        except Exception as e:
            logging.error(f"Error parsing mesh file: {e}")
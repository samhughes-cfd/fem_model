# pre_processing\parsing\mesh_parser.py

import logging
import numpy as np
import os
import re  # Import the regex module

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Set to DEBUG for detailed logs
    format='%(levelname)s: %(message)s'
)
# For more detailed logs during troubleshooting, uncomment one of the following:
# logging.getLogger().setLevel(logging.INFO)
# logging.getLogger().setLevel(logging.DEBUG)

def parse_mesh(mesh_file_path):
    """
    Parses a structured mesh file and computes element lengths using node coordinates.

    =============================
    Mesh Properties Mapping
    =============================

    Index   Property             Key in Dictionary         Data Type             Shape     Units  
    ------------------------------------------------------------------------------------------------
    0       Node IDs             `node_ids`               `np.ndarray[int]`      (N,)       -      
    1       Node Positions       `node_coordinates`       `np.ndarray[float]`    (N, 3)     [m] 
    2       Connectivity         `connectivity`           `np.ndarray[int]`      (M, 2)     -      
    3       Element IDs          `element_ids`            `np.ndarray[int]`      (M,)       -      
    4       Element Lengths      `element_lengths`        `np.ndarray[float]`    (M,)       [m] 
    5       Element Types        `element_types`          `np.ndarray[str]`      (M,)       -      

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
            - 'node_ids': np.ndarray[int]
            - 'node_coordinates': np.ndarray[float]
            - 'connectivity': np.ndarray[int]
            - 'element_lengths': np.ndarray[float]
            - 'element_ids': np.ndarray[int]
            - 'element_types': np.ndarray[str]

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
    - Nodes must be formatted as `ID X Y Z (Node1,Node2) ElementType`, where connectivity is optional.
    - If connectivity is missing, `-` is used as a placeholder.
    - Inline comments (#) are ignored.
    """

    # 1. Ensure the file exists
    if not os.path.exists(mesh_file_path):
        logging.error(f"[Mesh] File not found: {mesh_file_path}")
        raise FileNotFoundError(f"{mesh_file_path} not found")

    # 2. Initialize storage lists
    node_ids = []
    node_coordinates = []
    connectivity_list = []
    element_types = []

    current_section = None
    found_mesh_section = False

    # 3. Compile the regex pattern for efficiency
    mesh_line_pattern = re.compile(
        r'^\s*(\d+)\s+'                 # Node ID
        r'([\d\.eE+-]+)\s+'             # X-coordinate
        r'([\d\.eE+-]+)\s+'             # Y-coordinate
        r'([\d\.eE+-]+)\s+'             # Z-coordinate
        r'(\((\d+),(\d+)\)|-)\s+'       # Connectivity
        r'(.*)$'                        # Element Type
    )

    # 4. Read and parse the file
    with open(mesh_file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            # Remove inline comments and strip whitespace
            line = raw_line.split("#")[0].strip()

            # Log the current line being processed
            logging.debug(f"[Mesh] Line {line_number}: '{raw_line.strip()}'")

            # Skip empty lines
            if not line:
                logging.debug(f"[Mesh] Line {line_number} is empty. Skipping.")
                continue

            # Section detection
            if line.lower() == "[mesh]":
                logging.info(f"[Mesh] Found [mesh] section at line {line_number}.")
                current_section = "mesh"
                found_mesh_section = True
                continue

            # Process lines within the [mesh] section
            if current_section == "mesh":
                # Skip header line by matching known headers
                headers = ["node_ids", "x", "y", "z", "connectivity", "element_type"]
                if all(header in line.lower() for header in headers):
                    logging.debug(f"[Mesh] Line {line_number} is a header. Skipping.")
                    continue

                # Apply the regex pattern to the line
                match = mesh_line_pattern.match(line)
                if not match:
                    logging.warning(f"[Mesh] Line {line_number}: Line format invalid. Skipping line.")
                    continue

                # Extract fields using regex groups
                node_id_str = match.group(1)
                x_str = match.group(2)
                y_str = match.group(3)
                z_str = match.group(4)
                connectivity_full = match.group(5)
                start_node_str = match.group(6)
                end_node_str = match.group(7)
                element_type_str = match.group(8).strip()

                # Parse numerical fields
                try:
                    node_id = int(node_id_str)
                    x = float(x_str)
                    y = float(y_str)
                    z = float(z_str)
                except ValueError as e:
                    logging.warning(f"[Mesh] Line {line_number}: Invalid numerical data. {e}. Skipping line.")
                    continue

                # Append node data
                node_ids.append(node_id)
                node_coordinates.append((x, y, z))
                logging.debug(f"[Mesh] Line {line_number}: Parsed node {node_id} with coordinates ({x}, {y}, {z}).")

                # Parse connectivity
                if connectivity_full != "-":
                    try:
                        start_node = int(start_node_str)
                        end_node = int(end_node_str)
                        connectivity_list.append((start_node, end_node))
                        element_types.append(element_type_str)
                        logging.debug(f"[Mesh] Line {line_number}: Parsed connectivity ({start_node}, {end_node}) with element type '{element_type_str}'.")
                    except ValueError as e:
                        logging.warning(f"[Mesh] Line {line_number}: Invalid connectivity node IDs. {e}. Skipping connectivity.")
                else:
                    logging.debug(f"[Mesh] Line {line_number}: Node {node_id} has no connectivity. Skipping element type.")
                continue

            # If outside [mesh] section, ignore the line
            logging.debug(f"[Mesh] Line {line_number} is outside [mesh] section. Ignoring.")

    # 5. Validate that [mesh] section was found
    if not found_mesh_section:
        logging.error(f"[Mesh] No [mesh] section found in '{mesh_file_path}'. Returning empty arrays.")
        return {
            'node_ids': np.empty((0,), dtype=int),
            'node_coordinates': np.empty((0, 3), dtype=float),
            'connectivity': np.empty((0, 2), dtype=int),
            'element_lengths': np.empty((0,), dtype=float),
            'element_ids': np.empty((0,), dtype=int),
            'element_types': np.empty((0,), dtype=str)
        }

    # 6. Convert lists to NumPy arrays
    node_ids_array = np.array(node_ids, dtype=int)
    node_coordinates_array = np.array(node_coordinates, dtype=float)

    if connectivity_list:
        connectivity_array = np.array(connectivity_list, dtype=int)
        element_types_array = np.array(element_types, dtype=str)
    else:
        connectivity_array = np.empty((0, 2), dtype=int)
        element_types_array = np.empty((0,), dtype=str)

    # 7. Compute element lengths
    try:
        element_lengths_array = compute_element_lengths(connectivity_array, node_coordinates_array, node_ids_array)
    except ValueError as e:
        logging.error(f"[Mesh] {e}")
        # Depending on requirements, you can choose to halt execution or continue
        raise

    # 8. Generate element IDs starting from 1
    element_ids_array = np.arange(1, connectivity_array.shape[0] + 1, dtype=int)

    # 9. Assemble the mesh dictionary
    mesh_dictionary = {
        'node_ids': node_ids_array,                     # Index 0
        'node_coordinates': node_coordinates_array,     # Index 1
        'connectivity': connectivity_array,             # Index 2
        'element_ids': element_ids_array,               # Index 3
        'element_lengths': element_lengths_array,       # Index 4
        'element_types': element_types_array            # Index 5
    }

    # 10. Log summary
    logging.info(
        f"[Mesh] Successfully parsed {len(element_types_array)} elements from {len(node_ids_array)} nodes in '{mesh_file_path}'."
    )

    return mesh_dictionary

def compute_element_lengths(connectivity_array, node_coordinates_array, node_ids_array):
    """
    Computes element lengths using NumPy vectorization.

    Parameters
    ----------
    connectivity_array : np.ndarray[int]
        Shape (M, 2) - Each row contains [start_node_id, end_node_id].
    node_coordinates_array : np.ndarray[float]
        Shape (N, 3) - Each row contains [x, y, z] coordinates for a node.
    node_ids_array : np.ndarray[int]
        Shape (N,) - Unique node IDs in the mesh.

    Returns
    -------
    np.ndarray[float]
        Shape (M,) - Lengths of each element.
    
    Raises
    ------
    ValueError
        If a node ID in connectivity is not found in node_ids_array.
    """
    
    if connectivity_array.size == 0:
        logging.debug("[Mesh] No connectivity data provided. Returning empty element lengths array.")
        return np.empty((0,), dtype=float)

    # Convert node_ids_array into an index map (Vectorized)
    node_id_to_index = np.argsort(node_ids_array)  # Get sorted indices
    sorted_node_ids = node_ids_array[node_id_to_index]  # Sort node_ids_array

    # Find positions of connectivity node IDs in node_ids_array
    start_indices = np.searchsorted(sorted_node_ids, connectivity_array[:, 0])
    end_indices = np.searchsorted(sorted_node_ids, connectivity_array[:, 1])

    # Ensure that all indices are valid
    if np.any(sorted_node_ids[start_indices] != connectivity_array[:, 0]) or \
       np.any(sorted_node_ids[end_indices] != connectivity_array[:, 1]):
        raise ValueError("[Mesh] Some connectivity nodes are not found in the provided node list.")

    # Directly index node coordinates (Vectorized)
    start_coords = node_coordinates_array[node_id_to_index[start_indices]]
    end_coords = node_coordinates_array[node_id_to_index[end_indices]]

    # Compute Euclidean distances in a single operation
    element_lengths = np.linalg.norm(end_coords - start_coords, axis=1)

    logging.debug(f"[Mesh] Computed element lengths: {element_lengths}")

    return element_lengths

# Standalone execution for quick testing
if __name__ == "__main__":
    # Example usage
    test_file = r"jobs\job_0001\mesh.txt"  # Replace with your actual mesh file path

    if not os.path.exists(test_file):
        logging.error(f"Test file '{test_file}' not found. Please ensure the file exists.")
    else:
        try:
            mesh_dict = parse_mesh(test_file)
            print("\n-------------Parsed Mesh Data-------------")
            if mesh_dict['node_ids'].size > 0:
                print(f"-----Node IDs-----\n{mesh_dict['node_ids']}\n")
                print("Shape:", mesh_dict['node_ids'].shape)
                print(f"-----Node Coordinates-----\n{mesh_dict['node_coordinates']}\n")
                print("Shape:", mesh_dict['node_coordinates'].shape)
                print(f"-----Connectivity-----\n{mesh_dict['connectivity']}\n")
                print("Shape:", mesh_dict['connectivity'].shape)
                print(f"-----Element IDs-----\n{mesh_dict['element_ids']}\n")
                print("Shape:", mesh_dict['element_ids'].shape)
                print(f"-----Element Lengths-----\n{mesh_dict['element_lengths']}\n")
                print("Shape:", mesh_dict['element_lengths'].shape)
                print(f"-----Element Types-----\n{mesh_dict['element_types']}\n")
                print("Shape:", mesh_dict['element_types'].shape)

            else:
                print("No data parsed.")
        except Exception as e:
            logging.error(f"An error occurred while parsing the mesh file: {e}")
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

import logging
import numpy as np
import os
import re

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def parse_mesh(mesh_file_path):
    """
    Parses a structured mesh file and ensures all indices use 0-based indexing.

    =============================
    Mesh Properties Mapping
    =============================

    Property             Key in Dictionary         Data Type             Shape     Indexing    Units  
    ------------------------------------------------------------------------------------------------
    Node IDs             `node_ids`               `np.ndarray[int]`      (N,)      0-based      -      
    Node Positions       `node_coordinates`       `np.ndarray[float]`    (N, 3)    0-based      [m] 
    Connectivity         `connectivity`           `np.ndarray[int]`      (M, 2)    0-based      -      
    Element IDs          `element_ids`            `np.ndarray[int]`      (M,)      0-based      -      
    Element Lengths      `element_lengths`        `np.ndarray[float]`    (M,)      0-based      [m] 
    Element Types        `element_types`          `np.ndarray[str]`      (M,)      0-based      -      

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
            - 'node_ids': np.ndarray[int]         (0-based indexing)
            - 'node_coordinates': np.ndarray[float]  (0-based indexing)
            - 'connectivity': np.ndarray[int]     (0-based indexing)
            - 'element_ids': np.ndarray[int]      (0-based indexing)
            - 'element_lengths': np.ndarray[float]  (0-based indexing)
            - 'element_types': np.ndarray[str]    (0-based indexing)

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
    array([0, 1, 2, ...])

    Notes
    -----
    - Nodes must be formatted as `ID X Y Z (Node1,Node2) ElementType`, where connectivity is optional.
    - If connectivity is missing, `-` is used as a placeholder.
    - Inline comments (#) are ignored.
    """

    if not os.path.exists(mesh_file_path):
        logging.error(f"[Mesh] File not found: {mesh_file_path}")
        raise FileNotFoundError(f"{mesh_file_path} not found")

    node_ids = []
    node_coordinates = []
    connectivity_list = []
    element_types = []

    current_section = None
    found_mesh_section = False

    mesh_line_pattern = re.compile(
        r'^\s*(\d+)\s+'                 # Node ID (1-based)
        r'([\d\.eE+-]+)\s+'             # X-coordinate
        r'([\d\.eE+-]+)\s+'             # Y-coordinate
        r'([\d\.eE+-]+)\s+'             # Z-coordinate
        r'(\((\d+),(\d+)\)|-)\s+'       # Connectivity (1-based)
        r'(.*)$'                        # Element Type
    )

    with open(mesh_file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.split("#")[0].strip()
            logging.debug(f"[Mesh] Line {line_number}: '{raw_line.strip()}'")

            if not line:
                continue

            if line.lower() == "[mesh]":
                current_section = "mesh"
                found_mesh_section = True
                continue

            if current_section == "mesh":
                headers = ["node_ids", "x", "y", "z", "connectivity", "element_type"]
                if all(header in line.lower() for header in headers):
                    continue

                match = mesh_line_pattern.match(line)
                if not match:
                    logging.warning(f"[Mesh] Line {line_number}: Invalid format. Skipping.")
                    continue

                node_id = int(match.group(1)) - 1  # Convert to 0-based indexing
                x, y, z = float(match.group(2)), float(match.group(3)), float(match.group(4))

                node_ids.append(node_id)
                node_coordinates.append((x, y, z))

                if match.group(5) != "-":
                    start_node = int(match.group(6)) - 1  # Convert to 0-based
                    end_node = int(match.group(7)) - 1    # Convert to 0-based
                    connectivity_list.append((start_node, end_node))
                    element_types.append(match.group(8).strip())

    if not found_mesh_section:
        logging.error(f"[Mesh] No [mesh] section found in '{mesh_file_path}'. Returning empty data.")
        return {
            'node_ids': np.empty((0,), dtype=int),
            'node_coordinates': np.empty((0, 3), dtype=float),
            'connectivity': np.empty((0, 2), dtype=int),
            'element_lengths': np.empty((0,), dtype=float),
            'element_ids': np.empty((0,), dtype=int),
            'element_types': np.empty((0,), dtype=str)
        }

    # Convert lists to NumPy arrays
    node_ids_array = np.array(node_ids, dtype=int)
    node_coordinates_array = np.array(node_coordinates, dtype=float)
    connectivity_array = np.array(connectivity_list, dtype=int) if connectivity_list else np.empty((0, 2), dtype=int)
    element_types_array = np.array(element_types, dtype=str) if element_types else np.empty((0,), dtype=str)

    # Compute element lengths
    element_lengths_array = compute_element_lengths(connectivity_array, node_coordinates_array, node_ids_array)

    # Convert element IDs to 0-based
    element_ids_array = np.arange(connectivity_array.shape[0], dtype=int)

    mesh_dictionary = {
        'node_ids': node_ids_array, # 0-based
        'node_coordinates': node_coordinates_array,
        'connectivity': connectivity_array,
        'element_ids': element_ids_array,  # 0-based
        'element_lengths': element_lengths_array,
        'element_types': element_types_array
    }

    logging.info(f"[Mesh] Parsed {len(element_types_array)} elements from {len(node_ids_array)} nodes.")
    return mesh_dictionary

def compute_element_lengths(connectivity_array, node_coordinates_array, node_ids_array):
    """
    Computes element lengths using NumPy vectorization.

    Parameters
    ----------
    connectivity_array : np.ndarray[int]
        Shape (M, 2) - Each row contains [start_node_index, end_node_index] (0-based).
    node_coordinates_array : np.ndarray[float]
        Shape (N, 3) - Each row contains [x, y, z] coordinates for a node.
    node_ids_array : np.ndarray[int]
        Shape (N,) - Unique node indices in the mesh.

    Returns
    -------
    np.ndarray[float]
        Shape (M,) - Lengths of each element.
    
    Raises
    ------
    ValueError
        If a node index in connectivity is out of bounds.
    """

    if connectivity_array.size == 0:
        logging.debug("[Mesh] No connectivity data provided. Returning empty element lengths array.")
        return np.empty((0,), dtype=float)

    start_coords = node_coordinates_array[connectivity_array[:, 0]]
    end_coords = node_coordinates_array[connectivity_array[:, 1]]

    element_lengths = np.linalg.norm(end_coords - start_coords, axis=1)

    logging.debug(f"[Mesh] Computed element lengths: {element_lengths}")

    return element_lengths

def print_array_details(name, arr, index_range=None, is_coordinates=False, pair=False, one_based=False):
    """
    Enhanced function to print array details with indexing flexibility.
    
    Parameters
    ----------
    name : str
        Name of the array to be displayed.
    arr : np.ndarray
        The array whose details need to be printed.
    index_range : int, optional
        Expected range of indices for validation.
    is_coordinates : bool, optional
        Whether the array contains coordinate data.
    pair : bool, optional
        Whether the array contains paired connectivity data.
    one_based : bool, optional
        If True, displays indices in 1-based format (adjusted for printing only).
    """
    print(f"\n📌 len({name}) = {len(arr)}")

    if len(arr) == 0:
        print(f"❌ {name}: No data found.")
        return

    # Min and Max Values
    if is_coordinates and arr.ndim == 2 and arr.shape[1] >= 3:
        x_min, y_min, z_min = np.min(arr[:, :3], axis=0)
        x_max, y_max, z_max = np.max(arr[:, :3], axis=0)
        print(f"🔹 Min Value: ({x_min:.4f}, {y_min:.4f}, {z_min:.4f}), Max Value: ({x_max:.4f}, {y_max:.4f}, {z_max:.4f})")
    elif pair and arr.ndim == 2:
        min_pair = arr.min(axis=0)
        max_pair = arr.max(axis=0)
        print(f"🔹 Min Entry: {min_pair}, Max Entry: {max_pair}")
    elif np.issubdtype(arr.dtype, np.number):
        min_val = np.min(arr)
        max_val = np.max(arr)
        print(f"🔹 Min Value: {min_val}, Max Value: {max_val}")
    elif arr.dtype.kind in {'U', 'S', 'O'}:  # Handles string types
        unique_values = np.unique(arr)
        print(f"🔹 Unique Values ({len(unique_values)} total): {unique_values[:5]}{'...' if len(unique_values) > 5 else ''}")

    # Array Indices
    min_index = 0 if not one_based else 1
    max_index = (len(arr) - 1) if not one_based else len(arr)
    print(f"🔹 Min Index: {min_index}, Max Index: {max_index}")

    # Adjust for one-based printing
    arr_to_print = arr + 1 if one_based and arr.ndim == 1 else arr
    if one_based and pair and arr.ndim == 2:
        arr_to_print = arr + 1  # Adjust connectivity pairs to 1-based

    # Display Contents - Show first and last 5 elements if the list is long
    if len(arr) > 10:
        if arr.ndim == 1:
            print(f"📜 Entries:\n{arr_to_print[:5]} ... {arr_to_print[-5:]}")
        elif pair and arr.ndim == 2:
            print(f"📜 Entries:\n{arr_to_print[:5]} \n ... \n{arr_to_print[-5:]}")
        else:
            print(f"📜 Entries (truncated):\n{arr[:5]} \n ... \n{arr[-5:]}")
    else:
        print(f"📜 Entries:\n{arr_to_print}")

    print("-" * 40)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Example usage
    test_file = r"jobs\job_0001\mesh.txt"  # Replace with your actual mesh file path

    # Set indexing type: True for one-based, False for zero-based
    one_based_indexing = False  # Now using 0-based indexing

    if not os.path.exists(test_file):
        logging.error(f"Test file '{test_file}' not found. Please ensure the file exists.")
    else:
        try:
            mesh_dict = parse_mesh(test_file)

            print("\n" + "="*80)
            print("                          Parsed Mesh Data")
            print("="*80)

            # General mesh information
            num_nodes = len(mesh_dict.get('node_ids', []))
            num_elements = len(mesh_dict.get('element_ids', []))  # Ensure we're using element_ids

            print(f"🔹 Total Nodes (n): {num_nodes}")
            print(f"🔹 Total Elements (m): {num_elements}")
            print("-" * 80)

            # Ensure m = n - 1 condition
            if num_elements != num_nodes - 1:
                print(f"⚠ WARNING: Expected m = n - 1, but got {num_elements} elements for {num_nodes} nodes.")
                print("-" * 80)

            # NODE INFORMATION
            if num_nodes > 0:
                print("\nNODE INFORMATION")
                print("=" * 80)
                
                # Node IDs
                print_array_details(
                    name="node_ids",
                    arr=mesh_dict['node_ids'],
                    is_coordinates=False,
                    one_based=one_based_indexing
                )
                
                # Node Coordinates
                print_array_details(
                    name="node_coordinates",
                    arr=mesh_dict['node_coordinates'],
                    is_coordinates=True
                )

            # ELEMENT INFORMATION
            if num_elements > 0:
                print("\nELEMENT INFORMATION")
                print("=" * 80)
                
                # Connectivity
                print_array_details(
                    name="connectivity",
                    arr=mesh_dict['connectivity'],
                    pair=True,  # Connectivity contains pairs of node indices
                    one_based=one_based_indexing
                )
                
                # Element IDs
                print_array_details(
                    name="element_ids",
                    arr=mesh_dict['element_ids'],
                    is_coordinates=False,
                    one_based=one_based_indexing
                )
                
                # Element Lengths
                print_array_details(
                    name="element_lengths",
                    arr=mesh_dict['element_lengths'],
                    is_coordinates=False
                )
                
                # Element Types
                print_array_details(
                    name="element_types",
                    arr=mesh_dict['element_types'],
                    is_coordinates=False
                )

            print("\n✅ Mesh parsing completed successfully!")
            print("="*80)

        except Exception as e:
            logging.error(f"An error occurred while parsing the mesh file: {e}")
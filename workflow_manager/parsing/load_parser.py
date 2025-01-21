# workflow_manager/parsing/load_parser.py

import numpy as np
import logging

def parse_load(file_path):
    """
    Parses point load vectors from a .txt file and returns a 2D NumPy array.

    Each row corresponds to a point load and contains:
    [x, y, z, F_x, F_y, F_z, M_x, M_y, M_z]

    Parameters:
    - file_path (str): Path to the load input .txt file.

    Returns:
    - np.ndarray: 2D NumPy array of shape (n, 9), where n is the number of point loads.
    """
    loads_list = []
    found_header = False

    with open(file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            
            # Skip empty lines or comments
            if not line or line.startswith("#"):
                continue

            # Detect and skip header
            if not found_header and "[" in line and "]" in line:
                found_header = True
                logging.debug(f"Header found at line {line_number}.")
                continue

            # Process data lines
            if found_header:
                parts = line.split()

                if len(parts) != 9:
                    logging.warning(f"Line {line_number}: Expected 9 columns, found {len(parts)}. Skipping line.")
                    continue

                try:
                    # Convert all values to float
                    load_values = list(map(float, parts))
                    loads_list.append(load_values)
                    logging.debug(f"Line {line_number}: Parsed point load {load_values}.")
                    
                except ValueError as e:
                    logging.warning(f"Line {line_number}: Value conversion error: {e}. Skipping line.")
                    continue

    if not loads_list:
        logging.error(f"No valid load data found in file '{file_path}'. Returning empty array.")
        return np.empty((0, 9), dtype=float)  # Return an empty 2D array with 9 columns

    # Convert list to a 2D NumPy array
    loads_array = np.array(loads_list, dtype=float)

    logging.info(f"Total point loads parsed: {loads_array.shape[0]} from file '{file_path}'.")

    return loads_array
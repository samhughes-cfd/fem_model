# pre_processing\parsing\load_parser.py

import numpy as np
import logging
import re
import os

# Set up logging for debugging
logging.basicConfig(level=logging.WARNING)
#logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)

def parse_load(file_path):
    """
    Parses point load vectors from a structured text file and returns a 2D NumPy array.

    =============================
    Load Properties Mapping
    =============================

    Index   Property    Symbol     Units
    --------------------------------------
    0       X-Position  [x]        [m]
    1       Y-Position  [y]        [m]
    2       Z-Position  [z]        [m]
    3       Force X     [F_x]      [N]
    4       Force Y     [F_y]      [N]
    5       Force Z     [F_z]      [N]
    6       Moment X    [M_x]      [N·m]
    7       Moment Y    [M_y]      [N·m]
    8       Moment Z    [M_z]      [N·m]

    Only numerical values within the `[Loads]` section are processed. Empty lines 
    and comments (`#`) are ignored. Malformed rows are skipped with a warning.

    Parameters
    ----------
    file_path : str
        Path to the load input file.

    Returns
    -------
    numpy.ndarray
        A NumPy array of shape `(N, 9)`, where `N` is the number of valid point loads.
        Each row represents a point load with `[x, y, z, F_x, F_y, F_z, M_x, M_y, M_z]`.

    Raises
    ------
    ValueError
        If a load property cannot be converted to a float.

    Warnings
    --------
    Logs a warning if an invalid load property is encountered.
    """

    # Step 1: Check if the file exists
    if not os.path.exists(file_path):
        logging.error(f"[Load] File not found: {file_path}")
        raise FileNotFoundError(f"{file_path} not found")

    loads_list = []
    header_pattern = re.compile(r"^\[loads\]$", re.IGNORECASE)  # Matches ONLY [Loads]
    current_section = None

    # Step 2: Read and process file
    with open(file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.split("#")[0].strip()  # Remove inline comments

            if not line:
                continue  # Skip empty lines

            # Detect the `[Loads]` section
            if header_pattern.match(line):
                logging.info(f"[Load] Found [Loads] section at line {line_number}. Beginning to parse loads.")
                current_section = "loads"
                continue

            # Skip any data before [Loads] section
            if current_section != "loads":
                continue  

            # Step 3: Process valid data lines
            parts = line.split()
            if len(parts) != 9:
                logging.warning(f"[Load] Line {line_number}: Expected 9 values, found {len(parts)}. Skipping.")
                continue

            try:
                loads_list.append([float(x) for x in parts])
            except ValueError:
                logging.warning(f"[Load] Line {line_number}: Invalid numeric data. Skipping.")

    # Step 4: Handle case where no valid loads were found
    if not loads_list:
        logging.error(f"[Load] No valid load data found in '{file_path}'. Returning empty array.")
        return np.empty((0, 9), dtype=float)

    # Step 5: Convert to NumPy array and log results
    loads_array = np.array(loads_list, dtype=float)
    logging.info(f"[Load] Parsed {loads_array.shape[0]} load entries from '{file_path}'.")
    logging.debug(f"[Load] Final parsed array:\n{loads_array}")

    return loads_array

# Standalone execution for testing
if __name__ == "__main__":
    test_file = r"jobs\job_0001\load.txt"  # Use raw string for Windows paths
    if not os.path.exists(test_file):
        logging.error(f"Test file '{test_file}' not found. Make sure it exists before running.")
    else:
        try:
            output = parse_load(test_file)
            print("\n-------------Parsed Load Data-------------\n", output)
        except Exception as e:
            logging.error(f"Error parsing load file: {e}")
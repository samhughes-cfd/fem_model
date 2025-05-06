# pre_processing\parsing\distributed_load_parser.py

import numpy as np
import logging
import re
import os

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def parse_distributed_load(file_path):
    """
    Parses discrete distributed load vectors from a structured text file and returns a 2D NumPy array.
    """

    # Step 1: Check if the file exists
    if not os.path.exists(file_path):
        logging.error(f"[Distributed load] File not found: {file_path}")
        raise FileNotFoundError(f"{file_path} not found")

    logging.info(f"[Distributed load] Reading file: {file_path}")

    loads_list = []
    header_pattern = re.compile(r"^\[Distributed load\]$", re.IGNORECASE)  # Matches [Distributed load]
    current_section = None
    first_numeric_line_detected = False  # Track if header is skipped

    # Step 2: Read and process file
    with open(file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            logging.debug(f"Processing line {line_number}: {raw_line.strip()}")

            line = raw_line.split("#")[0].strip()  # Remove inline comments

            if not line:
                logging.debug(f"Skipping empty line {line_number}")
                continue  # Skip empty lines

            # Detect the `[Loads]` section
            if header_pattern.match(line):
                logging.info(f"[Distributed load] Found [Distributed load] section at line {line_number}. Beginning to parse loads.")
                current_section = "[Distributed load]"
                continue

            # Skip any data before [Distributed load] section
            if current_section != "[Distributed load]":
                logging.debug(f"Skipping line {line_number}: Outside [Distributed load] section")
                continue  

            # Step 3: Process valid data lines
            parts = line.split()

            # Log the parsed split parts
            logging.debug(f"Line {line_number} split into: {parts}")

            # If the first row contains any non-numeric characters, treat it as a header and skip
            if not first_numeric_line_detected:
                if any(re.search(r"[^\d\.\-+eE]", p) for p in parts):  
                    continue  # Skip this row if it contains anything non-numeric
                first_numeric_line_detected = True  # Set flag after processing first data row

            if len(parts) != 9:
                logging.warning(f"[Distributed load] Line {line_number}: Expected 9 values, found {len(parts)}. Content: {parts}. Skipping.")
                continue

            try:
                numeric_values = [float(x) for x in parts]
                loads_list.append(numeric_values)
                logging.debug(f"Successfully parsed line {line_number}: {numeric_values}")
            except ValueError as e:
                logging.warning(f"[Distributed load] Line {line_number}: Invalid numeric data '{parts}'. Error: {e}. Skipping.")

    # Step 4: Handle case where no `[Distributed load]` section was found
    if current_section is None:
        logging.warning("[Distributed load] WARNING: No [Distributed load] section detected! Parsing from first valid numeric row.")
        current_section = "[Distributed load]"  # Set manually to allow parsing

    # Step 5: Handle case where no valid loads were found
    if not loads_list:
        logging.error(f"[Distributed load] No valid load data found in '{file_path}'. Returning empty array.")
        return np.empty((0, 9), dtype=float)

    # Step 6: Convert to NumPy array and log results
    distributed_load_array = np.array(loads_list, dtype=float)
    logging.info(f"[Distributed load] Successfully parsed {distributed_load_array.shape[0]} load entries from '{file_path}'.")
    logging.debug(f"[Distributed load] Final parsed array:\n{distributed_load_array}")

    return distributed_load_array

# Standalone execution for testing
if __name__ == "__main__":
    test_file = r"jobs\job_0004\distributed_load.txt"  # Use raw string for Windows paths
    if not os.path.exists(test_file):
        logging.error(f"Test file '{test_file}' not found. Make sure it exists before running.")
    else:
        try:
            output = parse_distributed_load(test_file)
            print("\n-------------Parsed [Distributed load] Data-------------\n", output)
        except Exception as e:
            logging.error(f"Error parsing load file: {e}")
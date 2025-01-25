# pre_processing\parsing\materials_parser.py

import numpy as np
import logging
import re
import os

logging.basicConfig(level=logging.INFO)

def parse_material(file_path):
    """
    Parses material properties from a structured text file and returns them as a NumPy array.

    =============================
    Material Properties Mapping
    =============================

    Index   Property                            Symbol     Units
    --------------------------------------------------------------
    0       Young’s Modulus                     [E]        [Pa]     
    1       Shear Modulus                       [G]        [Pa]     
    2       Poisson’s Ratio                     [nu]       [-]      
    3       Density                             [rho]      [kg/m³]  

    Only values within the `[Material]` section are processed. The function skips empty lines 
    and comments (`#`) while parsing. Missing values are replaced with `NaN`.

    Parameters
    ----------
    file_path : str
        Path to the material properties file.

    Returns
    -------
    numpy.ndarray
        A NumPy array of shape `(1, 4)`, containing material properties `[E, G, nu, rho]`. 
        Missing properties are set to `NaN`.

    Raises
    ------
    ValueError
        If a property cannot be converted to a float.

    Warnings
    --------
    Logs a warning if an invalid material property is encountered.
    """

    # Step 1: Check if the file exists
    if not os.path.exists(file_path):
        logging.error(f"[Material] File not found: {file_path}")
        raise FileNotFoundError(f"{file_path} not found")

    # Step 2: Initialize material array with NaN
    material_array = np.full((1, 4), np.nan)

    # Define material properties mapping
    material_keys = ["E", "G", "nu", "rho"]
    material_map = {key: idx for idx, key in enumerate(material_keys)}

    current_section = None
    key_pattern = re.compile(r"\[(.*?)\]\s*(.*)")  # Match `[Key] Value` format
    found_material_section = False

    # Step 3: Read and process file
    with open(file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.split("#")[0].strip()  # Remove inline comments

            logging.debug(f"[Material] Processing line {line_number}: '{line}'")

            if not line:
                logging.debug(f"[Material] Line {line_number} is empty. Skipping.")
                continue  # Skip empty lines

            # Detect the `[Material]` section
            if line.lower() == "[material]":
                logging.info(f"[Material] Found [Material] section at line {line_number}.")
                current_section = "material"
                found_material_section = True
                continue

            if current_section != "material":
                logging.warning(f"[Material] Line {line_number} ignored: Outside [Material] section.")
                continue  

            # Step 4: Process `[Key] Value` pairs
            match = key_pattern.match(line)
            if match:
                key, value = match.groups()
                key = key.strip()

                if key in material_map:
                    try:
                        material_array[0, material_map[key]] = float(value.strip())
                        logging.debug(f"[Material] Parsed: {key} -> {value.strip()}")
                    except ValueError:
                        logging.warning(f"[Material] Line {line_number}: Invalid float value for {key}. Skipping.")

    # Step 5: Handle missing `[Material]` section
    if not found_material_section:
        logging.warning(f"[Material] No valid `[Material]` section found in '{file_path}'. Returning NaN-filled array.")

    logging.info(f"[Material] Parsed data from '{file_path}':\n{material_array}")

    return material_array

# Standalone execution for testing
if __name__ == "__main__":
    test_file = r"jobs\base\material.txt"  # Use raw string for Windows paths
    if not os.path.exists(test_file):
        logging.error(f"Test file '{test_file}' not found. Make sure it exists before running.")
    else:
        try:
            output = parse_material(test_file)
            print("\n📊 Parsed Material Data:\n", output)
        except Exception as e:
            logging.error(f"Error parsing material file: {e}")
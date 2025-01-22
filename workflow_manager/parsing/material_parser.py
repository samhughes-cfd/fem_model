# pre_processing\parsing\materials_parser.py

import numpy as np
import logging
import re

logging.basicConfig(level=logging.WARNING)

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

    Data Fetching
    -------------
    The returned `material_array` supports standard NumPy indexing techniques:

    Technique           Command                   Description
    -----------------------------------------------------------------
    Basic Indexing      `material_array[0, 0]`    Fetches Young’s Modulus [E]
    Slicing             `material_array[0, :3]`   Fetches `[E, G, nu]`
    Fancy Indexing      `material_array[0, [1,3]]`Fetches `[G, rho]`

    Example
    -------
    >>> material_data = parse_material("materials.txt")
    >>> print(material_data)
    array([[2.1e11, 8.0e10, 0.3, 7850.0]])

    Notes
    -----
    - Properties must be in `[Key] Value` format with keys enclosed in square brackets.
    - If a property is missing, its index will contain `NaN`.
    - Inline comments (text following `#`) are ignored.
    """

    # Initialize material array with NaN
    material_array = np.full((1, 4), np.nan)

    # Define material properties mapping
    material_keys = ["E", "G", "nu", "rho"]
    material_map = {key: idx for idx, key in enumerate(material_keys)}

    current_section = None
    key_pattern = re.compile(r"\[(.*?)\]\s*(.*)")  # Match `[Key] Value` format

    with open(file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.split("#")[0].strip()  # Remove inline comments

            if not line:
                continue  # Skip empty lines

            if line.lower() == "[material]":
                current_section = "material"
                continue

            if current_section != "material":
                continue  # Ignore all other sections

            match = key_pattern.match(line)
            if match:
                key, value = match.groups()
                key = key.strip()

                if key in material_map:
                    try:
                        material_array[0, material_map[key]] = float(value.strip())
                    except ValueError:
                        logging.warning(f"Invalid float value for {key} at line {line_number}: {value.strip()}")

    return material_array
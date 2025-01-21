# pre_processing\parsing\materials_parser.py

import numpy as np
import logging

def parse_material(file_path):
    """
    Parses material properties from a structured text file and returns them as a (1,4) NumPy array.

    The function reads material properties from a file, extracting values corresponding to the following indices:

    =============================
    Material Properties Mapping
    =============================

    Index   Property                            Symbol     Units
    --------------------------------------------------------------
    0       Young’s Modulus                     [E]        [Pa]     
    1       Shear Modulus                       [G]        [Pa]     
    2       Poisson’s Ratio                     [nu]       [-]      
    3       Density                             [rho]      [kg/m^3]  

    Only values within the `[Material]` section are processed. The function ignores empty lines and comments (`#`).

    Parameters
    ----------
    file_path : str
        Path to the material properties file.

    Returns
    -------
    numpy.ndarray
        A NumPy array of shape `(1,4)`, containing the parsed material properties `[E, G, nu, rho]`.
        If any property is missing in the input file, its corresponding index is set to `NaN`.

    Raises
    ------
    ValueError
        If a material property cannot be converted to a float.

    Warnings
    --------
    Logs a warning if an invalid material property is encountered.

    The function can be called as follows:

    >>> material_data = parse_material("materials.txt")

    Notes
    -----
    - The function assumes material properties are formatted as `[Key] Value`, with keys enclosed in square brackets.
    - If a material property is missing, `NaN` is assigned at the corresponding index.
    - Inline comments (after `#`) are ignored.
    """

    material_values = [np.nan, np.nan, np.nan, np.nan]  # Placeholder for [E, G, nu, rho]
    material_keys = ["E", "G", "nu", "rho"]
    current_section = None

    with open(file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue  # Skip empty lines and comments

            # Detect the [Material] section
            if line.lower().startswith("[material]"):
                current_section = "material"
                continue

            if current_section != "material":
                continue  # Ignore all other sections

            # Expect key-value pairs in bracketed format
            if line.startswith("[") and "]" in line:
                try:
                    key = line[1:line.index("]")].strip()  # Extract key inside brackets
                    remainder = line[line.index("]") + 1:].strip().split("#")[0].strip()  # Strip inline comments
                    value = float(remainder)  # Convert value to float
                    
                    if key in material_keys:
                        idx = material_keys.index(key)
                        material_values[idx] = value  # Assign to correct index
                except ValueError:
                    logging.warning(f"Line {line_number}: Invalid material property: '{raw_line}'")
                    continue

    return np.array([material_values], dtype=float)  # Return (1,4) array
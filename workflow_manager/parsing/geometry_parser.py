# pre_processing\parsing\geometry_parser.py

import numpy as np
import logging
import re

logging.basicConfig(level=logging.WARNING)

def parse_geometry(file_path):
    """
    Parses beam geometry and cross-sectional properties from a structured text file.
    Extracts properties from `[geometry]` and `[section_geometry]` sections.

    The function returns a NumPy array of shape `(1, 20)`, where missing values are set to NaN.

    ============================= 
    Geometry Properties Mapping
    =============================

    Index   Property                            Symbol     Units  
    ----------------------------------------------------------------
    0       Beam Length                         [L]        [m]     
    1       Cross-sectional Area                [A]        [m²]    
    2       Moment of Inertia (x-axis)          [I_x]       [m⁴]    
    3       Moment of Inertia (y-axis)          [I_y]       [m⁴]    
    4       Moment of Inertia (z-axis)          [I_z]       [m⁴]    
    5       Polar Moment of Inertia             [J]        [m⁴]    
    6       Torsional Constant                  [J_t]      [-]
    7       Warping Moment of Inertia           [I_w]      [m⁶]
    8       Centroid (x-position)               [c_x]      [m]     
    9       Centroid (y-position)               [c_y]      [m]     
    10      Centroid (z-position)               [c_z]      [m]     
    11      Static Moment (x-axis)              [s_x]      [m³]    
    12      Static Moment (y-axis)              [s_y]      [m³]    
    13      Static Moment (z-axis)              [s_z]      [m³]    
    14      Radius of Gyration (x-axis)         [r_x]      [m]     
    15      Radius of Gyration (y-axis)         [r_y]      [m]     
    16      Radius of Gyration (z-axis)         [r_z]      [m]
    17      Position of Shear Center (x-axis)   [x_s]      [m]          
    18      Position of Shear Center (y-axis)   [y_s]      [m]
    19      Position of Shear Center (z-axis)   [z_s]      [m]

    Only values within the `[Geometry]` and `[Section_Geometry]` sections are processed.
    Empty lines and comments (`#`) are ignored.

    Parameters
    ----------
    file_path : str
        Path to the geometry properties file.

    Returns
    -------
    numpy.ndarray
        A NumPy array of shape `(1,20)`, where missing values are assigned `NaN`.

    Raises
    ------
    ValueError
        If a geometry property cannot be converted to a float.

    Warnings
    --------
    Logs a warning if an invalid geometry property is encountered.

    Data Fetching
    -----------------------------
    The returned `geometry_array` supports various NumPy indexing techniques:

    Technique           Command                               Description                                  
    ---------------------------------------------------------------------------------------
    Basic Indexing      `geometry_array[0, 0]`                Fetches `L`                     
    Slicing             `geometry_array[0, :5]`               Fetches `[L, A, Ix, Iy, Iz]`               
    Fancy Indexing      `geometry_array[0, [8, 11, 17]]`      Fetches `[c_x, s_x, x_s]`                    

    Example:
    >>> geometry_data = parse_geometry("geometry.txt")
    >>> print(geometry_data)
    array([[8.0, 0.05, 1.3e-4, 2.4e-4, 3.5e-4, 5.1e-5, 4.1e-5, 2.2e-6,
            0.1, 0.2, 0.0, 0.15, 0.25, 0.35, 0.05, 0.07, 0.09, 0.12, 0.18, 0.0]])

    Notes
    -----
    - Properties are formatted as `[Key] Value`, with keys enclosed in square brackets.
    - If a property is missing, `NaN` is assigned at the corresponding index.
    - Inline comments (after `#`) are ignored.
    """

    geometry_array = np.full((1, 20), np.nan)

    geometry_keys = [
        "L", "A", "I_x", "I_y", "I_z", "J", "J_t", "I_w", 
        "c_x", "c_y", "c_z", "s_x", "s_y", "s_z", 
        "r_x", "r_y", "r_z", "x_s", "y_s", "z_s"
    ]
    geometry_map = {key: i for i, key in enumerate(geometry_keys)}

    relevant_sections = {"[geometry]", "[section_geometry]"}
    current_section = None

    key_pattern = re.compile(r"\[(.*?)\]\s*(.*)")  # Match `[Key] Value`

    with open(file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.split("#")[0].strip()  # Remove inline comments
            if not line:
                continue  # Skip empty lines

            if line in relevant_sections:
                current_section = "geometry"
                continue

            if current_section != "geometry":
                continue  # Ignore other sections

            match = key_pattern.match(line)
            if match:
                key, value = match.groups()
                key = key.strip()  # Preserve original case

                if key in geometry_map:
                    try:
                        geometry_array[0, geometry_map[key]] = float(value.strip())
                    except ValueError:
                        logging.warning(f"Invalid float value for {key} at line {line_number}: {value.strip()}")

    return geometry_array
    

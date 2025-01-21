# pre_processing\parsing\geometry_parser.py

import numpy as np
import logging

def parse_geometry(file_path):
    """
    Parses geometry and cross-sectional properties from a structured text file.

    This function extracts beam geometry and section properties from a file and stores them in a 
    fixed-order NumPy array of shape (1,20), where:

    =============================
    Geometry Properties Mapping
    =============================

    Index   Property                            Symbol     Units  
    ----------------------------------------------------------------
    0       Beam Length                         [L]        [m]     
    1       Cross-sectional Area                [A]        [m^2]    
    2       Moment of Inertia (x-axis)          [Ix]       [m^4]    
    3       Moment of Inertia (y-axis)          [Iy]       [m^4]    
    4       Moment of Inertia (z-axis)          [Iz]       [m^4]    
    5       Polar Moment of Inertia             [J]        [m^4]    
    6       Torsional Constant                  [J_t]      [-]
    7       Warping Moment of Inertia           [I_w]      [m^6]
    8       Centroid (x-position)               [c_x]      [m]     
    9       Centroid (y-position)               [c_y]      [m]     
    10      Centroid (z-position)               [c_z]      [m]     
    11      Static Moment (x-axis)              [s_x]      [m^3]    
    12      Static Moment (y-axis)              [s_y]      [m^3]    
    13      Static Moment (z-axis)              [s_z]      [m^3]    
    14      Radius of Gyration (x-axis)         [r_x]      [m]     
    15      Radius of Gyration (y-axis)         [r_y]      [m]     
    16      Radius of Gyration (z-axis)         [r_z]      [m]
    17      Position of Shear Center (x-axis)   [x_s]      [m]          
    18      Position of Shear Center (y-axis)   [y_s]      [m]
    19      Position of Shear Center (z-axis)   [z_s]      [m]

    Only values within the `[Geometry]` and `[Section_Geometry]` sections are processed. 
    The function ignores empty lines and comments (`#`).

    Parameters
    ----------
    file_path : str
        Path to the geometry properties file.

    Returns
    -------
    numpy.ndarray
        A NumPy array of shape `(1,20)`, containing the extracted geometry values.
        If any property is missing, its corresponding index is set to `NaN`.

    Raises
    ------
    ValueError
        If a geometry property cannot be converted to a float.

    Warnings
    --------
    Logs a warning if an invalid geometry property is encountered.

    Example Usage
    -------------
    >>> geometry_data = parse_geometry("geometry.txt")
    >>> print(geometry_data)
    array([[8.0, 0.05, 1.3e-4, 2.4e-4, 3.5e-4, 5.1e-5, 4.1e-5, 2.2e-6,
            0.1, 0.2, 0.0, 0.15, 0.25, 0.35, 0.05, 0.07, 0.09, 0.12, 0.18, 0.0]])

    Notes
    -----
    - The function assumes properties are formatted as `[Key] Value`, with keys enclosed in square brackets.
    - If a property is missing, `NaN` is assigned at the corresponding index.
    - Inline comments (after `#`) are ignored.
    - Parsing stops when `[Boundary_Conditions]` is encountered.
    """

    # Initialize geometry array with NaNs (ensures missing values are handled)
    geometry_values = [np.nan] * 20  

    # Ensure the order matches the documentation
    geometry_keys = [
        "L", "A", "Ix", "Iy", "Iz", "J", "J_t", "I_w", 
        "c_x", "c_y", "c_z", "s_x", "s_y", "s_z", 
        "r_x", "r_y", "r_z", "x_s", "y_s", "z_s"
    ]

    current_section = None

    with open(file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue  # Skip empty lines and comments

            # Detect relevant sections
            lower_line = line.lower()
            if lower_line.startswith("[geometry]") or lower_line.startswith("[section_geometry]"):
                current_section = "geometry"
                continue
            elif lower_line.startswith("[boundary_conditions]"):
                break  # Stop parsing when reaching boundary conditions

            if current_section != "geometry":
                continue  # Ignore all other sections

            # Expect bracketed key-value pairs
            if line.startswith("[") and "]" in line:
                try:
                    key = line[1:line.index("]")].strip()  # Extract key inside brackets
                    remainder = line[line.index("]") + 1:].strip().split("#")[0].strip()  # Strip inline comments
                    value = float(remainder)  # Convert value to float
                    
                    if key in geometry_keys:
                        idx = geometry_keys.index(key)
                        geometry_values[idx] = value  # Assign to correct index
                except ValueError:
                    logging.warning(f"Line {line_number}: Invalid geometry property: '{raw_line}'")
                    continue

    return np.array([geometry_values], dtype=float)  # Return (1,20) array
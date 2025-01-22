# workflow_manager/parsing/load_parser.py

import numpy as np
import logging
import re

logging.basicConfig(level=logging.WARNING)

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

    Data Fetching
    -------------
    The returned `loads_array` supports standard NumPy indexing techniques:

    Technique           Command                 Description
    -----------------------------------------------------------------
    Basic Indexing      `loads_array[0, 0]`     Fetches first point's `x`
    Slicing             `loads_array[:, :3]`    Fetches all `[x, y, z]` positions
    Fancy Indexing      `loads_array[:, [3,4]]` Fetches all `[F_x, F_y]` forces

    Example
    -------
    >>> loads = parse_load("loads.txt")
    >>> print(loads)
    array([[0.0, 1.2, 3.4, 500.0, 0.0, -300.0, 10.0, 5.0, 2.0],
           [2.3, 0.0, 1.5, -200.0, 600.0, 0.0, 0.0, -8.0, 1.5]])

    Notes
    -----
    - The function assumes the `[Loads]` section contains space-separated numerical values.
    - If a load row has missing or invalid values, it is skipped.
    - Inline comments (text following `#`) are ignored.
    """

    loads_list = []
    header_pattern = re.compile(r"\[.*?loads.*?\]", re.IGNORECASE)  # Match `[Loads]`
    current_section = None

    with open(file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.split("#")[0].strip()  # Remove inline comments

            if not line:
                continue  # Skip empty lines

            if header_pattern.match(line):  
                current_section = "loads"
                continue

            if current_section != "loads":
                continue  # Ignore all other sections

            parts = line.split()
            if len(parts) != 9:
                logging.warning(f"Line {line_number}: Expected 9 values, found {len(parts)}. Skipping line.")
                continue

            try:
                load_values = list(map(float, parts))
                loads_list.append(load_values)
            except ValueError:
                logging.warning(f"Line {line_number}: Invalid numeric data. Skipping line.")

    if not loads_list:
        logging.error(f"No valid load data found in '{file_path}'. Returning empty array.")
        return np.empty((0, 9), dtype=float)

    loads_array = np.array(loads_list, dtype=float)
    logging.info(f"Total point loads parsed: {loads_array.shape[0]} from file '{file_path}'.")

    return loads_array
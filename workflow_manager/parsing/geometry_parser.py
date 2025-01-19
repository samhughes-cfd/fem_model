# pre_processing\parsing\geometry_parser.py

import logging

def parse_geometry(file_path):
    """
    Parses geometry and section properties from a file.
    Returns a dictionary containing geometry-related values.
    """
    geometry = {}
    current_section = None

    with open(file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue  # Skip comments

            # Detect relevant sections
            lower_line = line.lower()
            if lower_line.startswith("[geometry]") or lower_line.startswith("[section_geometry]"):
                current_section = "geometry"
                continue
            elif lower_line.startswith("[boundary_conditions]"):
                current_section = None  # Stop parsing geometry

            if current_section != "geometry":
                continue  # Skip non-geometry sections

            # Expect bracketed key-value pairs
            if line.startswith("[") and "]" in line:
                try:
                    key = line[1:line.index("]")].strip()
                    remainder = line[line.index("]") + 1:].strip().split("#")[0].strip()  # Remove inline comments
                    value = float(remainder)
                    geometry[key] = value
                except ValueError:
                    logging.warning(f"Line {line_number}: Invalid geometry property: '{raw_line}'")
                    continue

    return geometry
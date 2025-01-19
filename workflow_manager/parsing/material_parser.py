# pre_processing\parsing\materials_parser.py

import logging

def parse_material(file_path):
    """
    Parses material properties from a file.
    Returns a dictionary with material parameters.
    """
    material = {}
    current_section = None

    with open(file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue  # Skip comments

            # Detect material section
            if line.lower().startswith("[material]"):
                current_section = "material"
                continue

            if current_section != "material":
                continue  # Ignore all other sections

            # Expect bracketed key-value pairs
            if line.startswith("[") and "]" in line:
                try:
                    key = line[1:line.index("]")].strip()
                    remainder = line[line.index("]") + 1:].strip().split("#")[0].strip()  # Strip inline comments
                    value = float(remainder)  # Convert to float
                    material[key] = value
                except ValueError:
                    logging.warning(f"Line {line_number}: Invalid material property: '{raw_line}'")
                    continue

    return material

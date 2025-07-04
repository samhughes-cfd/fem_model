# pre_processing\parsing\simulation_settings_parser.py

import os
import logging
import re

VALID_SIMULATION_TYPES = {"static", "modal", "dynamic"}

def parse_simulation_settings(file_path):
    """
    Parses simulation type from a structured simulation_settings.txt file.

    =============================
    Simulation Properties Mapping
    =============================

    Simulation Type    Description
    ----------------------------------------
    Static             Quasi-static structural analysis
    Dynamic            Time-dependent simulation (future use)
    Modal              Eigenvalue modal analysis (future use)

    The function reads a simulation configuration file and extracts the simulation type,
    validating it against the allowed set: {"static", "modal", "dynamic"}.

    Parameters
    ----------
    file_path : str
        Path to the simulation settings file.

    Returns
    -------
    dict
        Dictionary named `simulation_settings` with a single key `'type'`, containing one of:
        `'static'`, `'modal'`, or `'dynamic'`.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    ValueError
        If the simulation type is missing or unrecognized.

    Example
    -------
    File: simulation_settings.txt
        [Simulation]
        [Type]
        Static

    >>> simulation_settings = parse_simulation_settings("simulation_settings.txt")
    >>> print(simulation_settings)
    {'type': 'static'}

    Notes
    -----
    - The parser is case-insensitive for both headers and values.
    - Headers [Simulation] and [Type] must appear before the value.
    - Inline comments (`#`) are ignored.
    """
    if not os.path.exists(file_path):
        logging.error(f"Simulation settings file not found: {file_path}")
        raise FileNotFoundError(f"{file_path} not found")

    current_section = None
    type_found = None

    with open(file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            # Remove inline comments and strip whitespace
            line = raw_line.split("#")[0].strip()
            if not line:
                continue

            # Detect [Simulation] section
            if re.match(r"\[.*?simulation.*?\]", line, re.IGNORECASE):
                current_section = "simulation"
                continue

            # Detect [Type] section
            if current_section == "simulation" and re.match(r"\[.*?type.*?\]", line, re.IGNORECASE):
                current_section = "type"
                continue

            # Read the actual simulation type value
            if current_section == "type":
                sim_type = line.lower()
                if sim_type not in VALID_SIMULATION_TYPES:
                    logging.error(f"[Simulation] Line {line_number}: Invalid simulation type '{line}'. "
                                  f"Expected one of {list(VALID_SIMULATION_TYPES)}.")
                    raise ValueError(f"Invalid simulation type: '{line}'")
                type_found = sim_type
                break

    if not type_found:
        raise ValueError("Simulation type not specified in simulation_settings.txt")

    simulation_settings = {"type": type_found}
    return simulation_settings

# ------------------------------------------------
# Standalone execution for direct testing
# ------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    test_file = os.path.join("jobs", "base", "simulation_settings.txt")

    if not os.path.exists(test_file):
        logging.error(f"Test file '{test_file}' not found. Please ensure it exists before running.")
    else:
        try:
            simulation_settings = parse_simulation_settings(test_file)
            print("\n------------- Parsed Simulation Settings -------------\n")
            print(simulation_settings)
        except Exception as e:
            logging.error(f"❌ Error parsing simulation settings file: {e}")

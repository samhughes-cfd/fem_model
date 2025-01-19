# workflow_manager/parsing/load_parser.py

import numpy as np
import logging

def parse_load(file_path):
    """
    Parses load conditions and returns a NumPy array of shape (max_node_id, 6).
    """
    loads_dict = {}
    found_header = False

    with open(file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if not found_header and "[" in line and "]" in line:
                found_header = True
                continue

            parts = line.split()
            if len(parts) < 7:
                logging.warning(f"Line {line_number}: Incomplete data row: {raw_line}")
                continue

            try:
                node_id = int(parts[0])
                loads_dict[node_id] = [float(parts[i]) for i in range(1, 7)]
            except ValueError as e:
                logging.warning(f"Line {line_number}: Invalid numeric conversion: {parts} ({e})")
                continue

    if not loads_dict:
        return np.zeros((0, 6))

    max_node_id = max(loads_dict.keys())
    loads_array = np.zeros((max_node_id, 6), dtype=float)

    for node_id, vec in loads_dict.items():
        loads_array[node_id - 1, :] = vec

    return loads_array
# workflow_manager/parsing/boundary_condition_parser.py

import logging
import numpy as np

def parse_boundary_conditions(file_path):
    """
    Parses boundary conditions and returns a NumPy array of shape (max_node_id, 6).
    """
    bc_dict = {}
    current_section = None

    with open(file_path, 'r') as f:
        for line in f:
            raw_line = line.strip()
            if not raw_line or raw_line.startswith("#"):
                continue  # Ignore comments and blank lines

            if "[boundary_conditions]" in raw_line.lower():
                current_section = "boundary_conditions"
                continue

            if current_section != "boundary_conditions":
                continue  # Ignore all other sections

            if "[" in raw_line and "]" in raw_line:
                continue  # Skip headers like [node_ids] [constrained_dofs]

            parts = raw_line.split()
            if len(parts) < 7:
                logging.warning(f"Incomplete boundary condition line: '{raw_line}'")
                continue

            try:
                node_id = int(parts[0])
                bc_dict[node_id] = [int(v) for v in parts[1:7]]
            except ValueError as e:
                logging.warning(f"Invalid boundary condition format: '{raw_line}' => {e}")
                continue

    if not bc_dict:
        return np.zeros((0, 6), dtype=int)

    max_node_id = max(bc_dict.keys())
    boundary_array = np.zeros((max_node_id, 6), dtype=int)

    for node_id, cvals in bc_dict.items():
        boundary_array[node_id - 1, :] = cvals

    return boundary_array
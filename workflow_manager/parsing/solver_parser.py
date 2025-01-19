# workflow_manager\parsing\solver_parser.py

import logging
from processing.solver_registry import get_solver_registry

# Define all valid solver types
VALID_SOLVERS = {"Static", "Dynamic", "Modal"}

def parse_solver(file_path):
    """
    Parses solver configuration from the file and checks if the solver_name exists in the solver_registry.
    If a solver_name is missing in the registry, it is marked as "Off".

    Returns:
    - A dictionary: {'Static': 'Direct Solver', 'Dynamic': 'Off', 'Modal': 'Off'}
    """
    solver_registry = get_solver_registry()  # Load available solvers
    solver_data = {solver: "Off" for solver in VALID_SOLVERS}  # Default all solvers to "Off"
    found_header = False  # Ensure we skip the header

    with open(file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue  # Skip blank lines and comments

            # Remove inline comments
            if '#' in line:
                line = line.split('#', 1)[0].strip()

            # Detect header and skip it
            if not found_header:
                if "[" in line and "]" in line:
                    found_header = True  # Now look for actual data in the next rows
                    continue
                else:
                    logging.warning(f"Line {line_number}: Expected header row, got '{raw_line}'")
                    continue

            # Process solver type and optional solver name
            parts = line.split()
            if len(parts) < 1:
                logging.warning(f"Line {line_number}: Missing solver type.")
                continue

            solver_type = parts[0]  # Extract solver type

            # Validate solver type
            if solver_type not in VALID_SOLVERS:
                logging.error(f"Invalid solver type '{solver_type}'. Expected one of {VALID_SOLVERS}.")
                raise ValueError(f"Invalid solver type: {solver_type}")

            # Extract solver_name if specified
            solver_name = " ".join(parts[1:]) if len(parts) > 1 else None

            # If solver_name is missing, log a clearer warning
            if solver_name is None:
                if solver_type != "Static":
                    logging.info(f"Solver '{solver_type}' has no assigned solver name. Marking as 'Off'.")
                solver_data[solver_type] = "Off"
                continue

            # Check if the solver_name exists in the registry
            if solver_name in solver_registry:
                solver_data[solver_type] = solver_name  # Use registered solver
            else:
                logging.warning(f"Solver '{solver_type}' specified but solver name '{solver_name}' is not recognized. Setting to 'Off'.")
                solver_data[solver_type] = "Off"  # Mark as "Off" if not found in the registry

    return solver_data
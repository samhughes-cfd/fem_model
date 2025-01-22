# workflow_manager\parsing\solver_parser.py

import logging
import numpy as np
import re
from processing.solver_registry import get_solver_registry

logging.basicConfig(level=logging.WARNING)

# Define all valid solver types
VALID_SOLVERS = np.array(["Static", "Dynamic", "Modal"], dtype=str)

def parse_solver(file_path):
    """
    Parses solver configuration from a structured text file and validates against the solver registry.

    =============================
    Solver Properties Mapping
    =============================

    Index   Solver Type    Status         Description
    -------------------------------------------------------------
    0       Static        Solver Name    Direct solver for static problems
    1       Dynamic       Solver Name    Time-dependent solver
    2       Modal         Solver Name    Solver for eigenvalue problems

    The function reads a solver configuration file and checks if the specified solver names exist 
    in the solver registry. If a solver is not found, it is marked as `"Off"`.

    Parameters
    ----------
    file_path : str
        Path to the solver configuration file.

    Returns
    -------
    np.ndarray[str]
        A NumPy array of shape `(3,)`, containing solver names for `["Static", "Dynamic", "Modal"]`.
        If a solver is missing or unrecognized, `"Off"` is assigned.

    Raises
    ------
    ValueError
        If an invalid solver type is encountered.

    Warnings
    --------
    Logs a warning if a solver is unrecognized or missing.

    Data Fetching
    -------------
    The returned `solver_array` supports standard NumPy indexing techniques:

    Technique                Command                        Description
    -------------------------------------------------------------------
    Fetch solver for Static  `solver_array[0]`             Returns solver name for Static
    Fetch all solvers        `solver_array[:]`             Returns all solver names

    Example
    -------
    >>> solver_array = parse_solver("solver_config.txt")
    >>> print(solver_array)
    array(['Direct Solver', 'Off', 'Eigen Solver'], dtype='<U20')

    Notes
    -----
    - Solvers must be formatted as `SolverType SolverName` in the configuration file.
    - If no solver name is specified for a type, it is marked as `"Off"`.
    - Inline comments (`#`) are ignored.
    """

    solver_registry = get_solver_registry()  # Load available solvers
    solver_array = np.full((3,), "Off", dtype='<U20')  # Default all solvers to "Off"

    header_pattern = re.compile(r"\[.*?solver.*?\]", re.IGNORECASE)  # Match `[Solver]` section
    current_section = None

    with open(file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.split("#")[0].strip()  # Remove inline comments

            if not line:
                continue  # Skip empty lines

            if header_pattern.match(line):
                current_section = "solver"
                continue

            if current_section != "solver":
                continue  # Ignore all other sections

            parts = line.split()
            if len(parts) < 1:
                logging.warning(f"Line {line_number}: Missing solver type.")
                continue

            solver_type = parts[0].strip()  # Extract solver type

            # Validate solver type
            idx = np.where(VALID_SOLVERS == solver_type)[0]
            if len(idx) == 0:
                logging.error(f"Invalid solver type '{solver_type}'. Expected one of {VALID_SOLVERS.tolist()}.")
                raise ValueError(f"Invalid solver type: {solver_type}")

            # Extract solver_name if specified
            solver_name = " ".join(parts[1:]) if len(parts) > 1 else None

            if solver_name is None:
                logging.info(f"Solver '{solver_type}' has no assigned solver name. Marking as 'Off'.")
                continue

            # Validate solver name
            if solver_name in solver_registry:
                solver_array[idx[0]] = solver_name  # Use registered solver
            else:
                logging.warning(f"Solver '{solver_type}' specified but solver name '{solver_name}' is not recognized. Setting to 'Off'.")

    return solver_array
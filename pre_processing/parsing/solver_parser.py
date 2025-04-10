# pre_processing\parsing\solver_parser.py

import os
import logging
import numpy as np
import re

# Define all valid solver types
VALID_SOLVERS = np.array(["Static", "Dynamic", "Modal"], dtype=str)

def get_solver_registry():
    """
    Returns a registry of SciPy solvers available for solving linear systems.

    Returns:
        dict: Mapping solver names (str) to functions.
    """
    from scipy.sparse.linalg import cg, gmres, minres, bicg, bicgstab, lsmr, lsqr, spsolve
    from scipy.linalg import solve, lu_factor, lu_solve

    return {
        "direct_solver_dense": solve,
        "lu_decomposition_solver": lambda A, b: lu_solve(lu_factor(A), b),
        "direct_solver_sparse": spsolve,
        "conjugate_gradient_solver": cg,
        "generalized_minimal_residual_solver": gmres,
        "minimum_residual_solver": minres,
        "bi-conjugate_gradient_solver": bicg,
        "bi-conjugate_gradient_stabilized_solver": bicgstab,
        "least_squares_solver": lsmr,
        "sparse_least_squares_solver": lsqr,
    }

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

    # Load available solvers from the registry
    solver_registry = get_solver_registry()

    # Initialize all solver types to "Off" by default with increased string length
    solver_array = np.full((3,), "Off", dtype='<U30')  # Increased from '<U20' to '<U30'

    # Regex to detect the `[Solver]` section (case-insensitive)
    header_pattern = re.compile(r"\[.*?solver.*?\]", re.IGNORECASE)

    current_section = None

    if not os.path.exists(file_path):
        logging.error(f"Solver configuration file not found: {file_path}")
        raise FileNotFoundError(f"{file_path} not found")

    with open(file_path, 'r') as f:
        for line_number, raw_line in enumerate(f, 1):
            # Strip inline comments ('#') and whitespace
            line = raw_line.split("#")[0].strip()
            if not line:
                continue  # Skip empty lines

            # Detect [Solver] section
            if header_pattern.match(line):
                current_section = "solver"
                continue

            # If not in [Solver] section yet, ignore
            if current_section != "solver":
                continue

            # Parse each solver line: "Static SolverName", etc.
            parts = line.split()
            if len(parts) < 1:
                logging.warning(f"[Solver] Line {line_number}: Missing solver type.")
                continue

            solver_type = parts[0].strip()
            # Find which valid solver index this corresponds to
            idx = np.where(VALID_SOLVERS == solver_type)[0]
            if len(idx) == 0:
                logging.error(f"[Solver] Line {line_number}: Invalid solver type '{solver_type}'. "
                              f"Expected one of {VALID_SOLVERS.tolist()}.")
                raise ValueError(f"Invalid solver type: {solver_type}")

            # If there's a solver name, it will be the rest of the tokens
            solver_name = " ".join(parts[1:]) if len(parts) > 1 else None

            if not solver_name:
                # No solver name provided
                logging.info(f"[Solver] Line {line_number}: '{solver_type}' has no solver name. Marking as 'Off'.")
                continue

            # Validate solver_name against the registry
            if solver_name in solver_registry:
                solver_array[idx[0]] = solver_name
            else:
                logging.warning(f"[Solver] Line {line_number}: Unrecognized solver name '{solver_name}' for '{solver_type}'. "
                                "Setting to 'Off' (default).")

    return solver_array

# ------------------------------------------------
# Standalone execution for direct testing
# ------------------------------------------------
if __name__ == "__main__":
    # Directly parse the existing file at jobs\base\solver.txt
    test_file = r"jobs\base\solver.txt"  # Use raw string for Windows paths

    if not os.path.exists(test_file):
        logging.error(f"Test file '{test_file}' not found. Please ensure it exists before running.")
    else:
        try:
            solver_array = parse_solver(test_file)
            print("\n-------------Parsed Solver Data-------------\n", solver_array)
        except Exception as e:
            logging.error(f"Error parsing solver file: {e}")
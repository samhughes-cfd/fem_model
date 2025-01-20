"""
Automated Tests for FEM Input Parsers using `pytest`.

Run this test from the **project root directory**:

    python -m pytest pytest_testing/test_parsing.py -v --log-cli-level=INFO

All logs will be stored in `pytest_testing/logs/pytest_parsing.log`.
"""

import pytest
import os
import logging
import sys
import pprint
import numpy as np
import glob

# Ensure the project root is added to `sys.path`
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Define directories
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")  # Changed from PROJECT_ROOT to SCRIPT_DIR
JOBS_DIR = os.path.join(PROJECT_ROOT, "jobs")
BASE_DIR = os.path.join(JOBS_DIR, "base")

# Ensure the logs directory exists within pytest_testing
os.makedirs(LOG_DIR, exist_ok=True)

# Logging Configuration
LOG_FILE = os.path.join(LOG_DIR, "pytest_parsing.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),  # Log to file
        logging.StreamHandler(sys.stdout)  # Print to terminal
    ]
)

def debug_print(data, name="Data"):
    """Print and log parsed data structures."""
    logging.info(f"--- {name} ---")
    logging.info(pprint.pformat(data))  # Pretty-print to log
    print(f"\n--- {name} ---")
    pprint.pprint(data)  # Pretty-print to terminal

# Import parsers from `workflow_manager`
from workflow_manager.parsing.parser_base import ParserBase
from workflow_manager.parsing.geometry_parser import parse_geometry
from workflow_manager.parsing.material_parser import parse_material
from workflow_manager.parsing.mesh_parser import parse_mesh
from workflow_manager.parsing.load_parser import parse_load
from workflow_manager.parsing.solver_parser import parse_solver
from workflow_manager.parsing.boundary_condition_parser import parse_boundary_conditions

@pytest.fixture(scope="session")
def parser():
    """Provides a ParserBase instance for all tests."""
    return ParserBase()

@pytest.fixture(scope="session")
def base_dir_fixture():
    """Provides the base directory path."""
    return BASE_DIR

@pytest.fixture(scope="session")
def job_dirs_fixture():
    """Provides a list of all Job_* directory paths."""
    # Use glob to find all directories starting with 'Job_' inside JOBS_DIR
    job_dirs = glob.glob(os.path.join(JOBS_DIR, "Job_*"))
    if not job_dirs:
        pytest.fail(f"No Job_* directories found in {JOBS_DIR}")
    return job_dirs

@pytest.fixture
def validate_file():
    """Check if a file exists before parsing."""
    def _validate(file_path):
        if not os.path.isfile(file_path):
            pytest.fail(f"File not found: {file_path}")
        return file_path
    return _validate

@pytest.fixture(scope="session")
def log_file_path():
    """Provides the log file path."""
    return LOG_FILE

### ✅ **PHASE 1: Test individual parsing functions separately**
@pytest.mark.parametrize("parser_func, file_info, expected_type, additional_checks", [
    (
        parse_geometry,
        {"files": ["geometry.txt"], "directories": ["base"]},
        dict,
        []
    ),
    (
        parse_material,
        {"files": ["material.txt"], "directories": ["base"]},
        dict,
        []
    ),
    (
        parse_solver,
        {"files": ["solver.txt"], "directories": ["base"]},
        dict,
        []
    ),
    (
        parse_boundary_conditions,
        {"files": ["geometry.txt"], "directories": ["base"]},  # Boundary conditions are within geometry.txt
        np.ndarray,
        []
    ),
    (
        parse_load,
        {"files": ["load.txt"], "directories": ["job_dirs"]},  # 'job_dirs' will be expanded in the test
        np.ndarray,
        []
    ),
    (
        parse_mesh,
        {"files": ["mesh.txt"], "directories": ["job_dirs"]},  # Assuming mesh.txt is in Job_* directories
        dict,
        ["connectivity", "node_positions"]
    ),
])
def test_individual_parsers(parser_func, file_info, expected_type, additional_checks, validate_file, base_dir_fixture, job_dirs_fixture):
    """Test individual parsing functions and print/log results."""
    try:
        files = file_info["files"]
        directories = file_info["directories"]
        
        # Handle cases where directories include 'job_dirs'
        resolved_paths = []
        for idx, file in enumerate(files):
            dir_spec = directories[idx]
            if dir_spec == "base":
                directory = base_dir_fixture
                full_path = os.path.join(directory, file)
                full_path = validate_file(full_path)
                resolved_paths.append(full_path)
            elif dir_spec == "job_dirs":
                # Iterate through all Job_* directories for each file
                for job_dir in job_dirs_fixture:
                    full_path = os.path.join(job_dir, file)
                    if os.path.isfile(full_path):
                        validated_path = validate_file(full_path)
                        resolved_paths.append(validated_path)
                    else:
                        logging.warning(f"Expected file '{file}' not found in '{job_dir}'. Skipping.")
            else:
                pytest.fail(f"Unknown directory specification: {dir_spec}")

        if not resolved_paths:
            pytest.skip(f"No files found for parser '{parser_func.__name__}'")

        # For parse_mesh, each job directory has its own mesh file
        if parser_func == parse_mesh:
            for mesh_path in resolved_paths:
                # Assuming parse_mesh requires mesh file and geometry.txt from base
                geometry_path = os.path.join(base_dir_fixture, "geometry.txt")
                geometry_path = validate_file(geometry_path)
                parsed_data = parser_func(mesh_path, geometry_path)
                
                # Print & log parsed data structure
                debug_print(parsed_data, name=f"{parser_func.__name__} Output for '{mesh_path}'")
                
                # Check data types based on expected output
                assert isinstance(parsed_data, expected_type), (
                    f"Expected type '{expected_type.__name__}' for '{parser_func.__name__}' output, "
                    f"got '{type(parsed_data).__name__}'"
                )
                
                # Additional checks (e.g., required keys in dicts)
                for check in additional_checks:
                    if isinstance(parsed_data, dict):
                        assert check in parsed_data, f"Mesh data missing '{check}' key"
                
                logging.info(f"✅ '{parser_func.__name__}' successfully parsed mesh file '{mesh_path}'")
            return  # Mesh tests are handled per job directory

        # For parse_load and others, handle multiple files
        for file_path in resolved_paths:
            # Special handling if parse_boundary_conditions
            if parser_func == parse_boundary_conditions:
                # Boundary conditions are within geometry.txt
                parsed_data = parser_func(file_path)
            else:
                parsed_data = parser_func(file_path)

            # Print & log parsed data structure
            debug_print(parsed_data, name=f"{parser_func.__name__} Output for '{file_path}'")

            # Check data types based on expected output
            assert isinstance(parsed_data, expected_type), (
                f"Expected type '{expected_type.__name__}' for '{parser_func.__name__}' output, "
                f"got '{type(parsed_data).__name__}'"
            )

            # Additional checks (e.g., required keys in dicts)
            for check in additional_checks:
                if isinstance(parsed_data, dict):
                    assert check in parsed_data, f"Data missing '{check}' key"

            logging.info(f"✅ '{parser_func.__name__}' successfully parsed file '{file_path}'")

    except AssertionError as ae:
        logging.error(f"Assertion error during '{parser_func.__name__}' parsing: {ae}")
        raise
    except Exception as e:
        logging.error(f"Error during '{parser_func.__name__}' parsing: {e}")
        raise

### ✅ **PHASE 2: Test ParserBase class separately**
@pytest.mark.parametrize("key, file_path, expected_type, directory", [
    ("geometry", "geometry.txt", dict, "base"),
    ("material", "material.txt", dict, "base"),
    ("solver", "solver.txt", dict, "base"),
    ("load", "load.txt", np.ndarray, "job_dirs"),  # 'job_dirs' will be expanded in the test
    ("boundary_conditions", "geometry.txt", np.ndarray, "base"),  # Boundary conditions within geometry.txt
])
def test_parser_base(key, file_path, expected_type, directory, parser, validate_file, base_dir_fixture, job_dirs_fixture):
    """Test `ParserBase` class and print/log results."""
    try:
        # Determine full file paths
        if directory == "base":
            dir_path = base_dir_fixture
            full_path = os.path.join(dir_path, file_path)
            full_path = validate_file(full_path)
            
            if key == "boundary_conditions":
                # Boundary conditions are within geometry.txt
                parsed_data = parser.boundary_conditions_parser(full_path)
            else:
                # Other parsers
                parse_method = getattr(parser, f"{key}_parser")
                parsed_data = parse_method(full_path)
            
            # Print & log parsed data structure
            debug_print(parsed_data, name=f"{key} Parser Output (ParserBase) for '{full_path}'")

            # Check data type
            assert isinstance(parsed_data, expected_type), (
                f"Expected type '{expected_type.__name__}' for '{key}' parser output, "
                f"got '{type(parsed_data).__name__}'"
            )

            logging.info(f"✅ '{key}' parser in `ParserBase` successfully parsed '{file_path}'")

        elif directory == "job_dirs":
            # Iterate through all Job_* directories
            for job_dir in job_dirs_fixture:
                full_path = os.path.join(job_dir, file_path)
                if not os.path.isfile(full_path):
                    logging.warning(f"Expected file '{file_path}' not found in '{job_dir}'. Skipping.")
                    continue

                # Retrieve the parser method dynamically
                parse_method = getattr(parser, f"{key}_parser")

                # Parse the file
                parsed_data = parse_method(full_path)

                # Print & log parsed data structure
                debug_print(parsed_data, name=f"{key} Parser Output (ParserBase) for '{full_path}'")

                # Check data type
                assert isinstance(parsed_data, expected_type), (
                    f"Expected type '{expected_type.__name__}' for '{key}' parser output, "
                    f"got '{type(parsed_data).__name__}'"
                )

                logging.info(f"✅ '{key}' parser in `ParserBase` successfully parsed '{full_path}'")

    except AssertionError as ae:
        logging.error(f"Assertion error during '{key}' parsing: {ae}")
        raise
    except Exception as e:
        logging.error(f"Error during '{key}' parsing: {e}")
        raise

### ✅ **PHASE 3: Compare direct function outputs with ParserBase outputs**
@pytest.mark.parametrize("key, file_path, directory", [
    ("geometry", "geometry.txt", "base"),
    ("material", "material.txt", "base"),
    ("solver", "solver.txt", "base"),
    ("load", "load.txt", "job_dirs"),
    ("boundary_conditions", "geometry.txt", "base"),  # Boundary conditions within geometry.txt
])
def test_parser_base_consistency(key, file_path, directory, parser, validate_file, base_dir_fixture, job_dirs_fixture):
    """Ensure individual parsers produce the same output as ParserBase."""
    try:
        # Determine full file paths
        if directory == "base":
            dir_path = base_dir_fixture
            full_path = os.path.join(dir_path, file_path)
            full_path = validate_file(full_path)

            if key == "boundary_conditions":
                # Boundary conditions are within geometry.txt
                direct_output = parse_boundary_conditions(full_path)
                parser_base_output = parser.boundary_conditions_parser(full_path)
            else:
                # Other parsers
                direct_parser_func = globals().get(f"parse_{key}")
                if not direct_parser_func:
                    pytest.fail(f"No direct parser function found for '{key}'")
                parser_base_func = getattr(parser, f"{key}_parser")
                direct_output = direct_parser_func(full_path)
                parser_base_output = parser_base_func(full_path)

            # Compare outputs based on type
            if isinstance(direct_output, np.ndarray):
                assert np.array_equal(direct_output, parser_base_output), f"Inconsistent output for '{key}'"
            elif isinstance(direct_output, dict):
                assert direct_output == parser_base_output, f"Inconsistent output for '{key}'"
            else:
                assert direct_output == parser_base_output, f"Inconsistent output for '{key}'"

            logging.info(f"✅ '{key}': Direct function and `ParserBase` output match for '{full_path}'")

        elif directory == "job_dirs":
            # Iterate through all Job_* directories
            for job_dir in job_dirs_fixture:
                full_path = os.path.join(job_dir, file_path)
                if not os.path.isfile(full_path):
                    logging.warning(f"Expected file '{file_path}' not found in '{job_dir}'. Skipping.")
                    continue

                # Retrieve the direct parser function and ParserBase method
                if key == "boundary_conditions":
                    # Boundary conditions are within geometry.txt
                    geometry_path = os.path.join(base_dir_fixture, "geometry.txt")
                    geometry_path = validate_file(geometry_path)
                    direct_output = parse_boundary_conditions(geometry_path)
                    parser_base_output = parser.boundary_conditions_parser(geometry_path)
                else:
                    direct_parser_func = globals().get(f"parse_{key}")
                    if not direct_parser_func:
                        pytest.fail(f"No direct parser function found for '{key}'")
                    parser_base_func = getattr(parser, f"{key}_parser")
                    direct_output = direct_parser_func(full_path)
                    parser_base_output = parser_base_func(full_path)

                # Compare outputs based on type
                if isinstance(direct_output, np.ndarray):
                    assert np.array_equal(direct_output, parser_base_output), f"Inconsistent output for '{key}' in '{full_path}'"
                elif isinstance(direct_output, dict):
                    assert direct_output == parser_base_output, f"Inconsistent output for '{key}' in '{full_path}'"
                else:
                    assert direct_output == parser_base_output, f"Inconsistent output for '{key}' in '{full_path}'"

                logging.info(f"✅ '{key}': Direct function and `ParserBase` output match for '{full_path}'")

    except AssertionError as ae:
        logging.error(f"Assertion error in consistency check for '{key}': {ae}")
        raise
    except Exception as e:
        logging.error(f"Error comparing outputs for '{key}': {e}")
        raise
# testing/tests/pytest_parsing.py

"""
Centralized Tests for FEM Input Parsers using `pytest`.

Run these tests from the **project root directory**:
    python -m pytest testing/tests/pytest_parsing.py -v --log-cli-level=INFO

All logs will be stored in `testing/logs/pytest_parsing.log`.
"""

import pytest
import os
import pprint
import numpy as np
import glob
import logging

# === Setup Logging System ===

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../logs")
LOG_FILE = os.path.join(LOG_DIR, "pytest_parsing.log")

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Create logger
logger = logging.getLogger("pytest_parsing")
logger.setLevel(logging.INFO)

# Create File Handler (logs to file)
file_handler = logging.FileHandler(LOG_FILE, mode="w")  # Overwrites log each time
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Create Console Handler (logs to console)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Attach handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Initial log message
logger.info(f"🔥 LOGGING STARTED: Logs will be written to {LOG_FILE}")

# === Import Parsers ===
from workflow_orchestrator.parsing.base_parser import ParserBase
from workflow_orchestrator.parsing.geometry_parser import parse_geometry
from workflow_orchestrator.parsing.material_parser import parse_material
from workflow_orchestrator.parsing.mesh_parser import parse_mesh
from workflow_orchestrator.parsing.load_parser import parse_load
from workflow_orchestrator.parsing.solver_parser import parse_solver
from workflow_orchestrator.parsing.boundary_condition_parser import parse_boundary_conditions

# === Helper Function ===
def debug_print(data, name="Data"):
    """Print parsed data structures to console and log file."""
    log_message = f"\n--- {name} ---\n{pprint.pformat(data)}"
    print(log_message)
    logger.info(log_message)

# === Fixtures ===
@pytest.fixture(scope="session")
def parser():
    """Provides a ParserBase instance for all tests."""
    return ParserBase()

@pytest.fixture(scope="session")
def base_dir_fixture():
    """Provides the base directory path for test files."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../jobs/base")

@pytest.fixture(scope="session")
def job_dirs_fixture():
    """Finds all job directories dynamically."""
    jobs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../jobs")
    job_dirs = glob.glob(os.path.join(jobs_dir, "job_*"))
    if not job_dirs:
        pytest.fail(f"No job_* directories found in {jobs_dir}")
    return job_dirs

@pytest.fixture
def validate_file():
    """Ensures a file exists before testing."""
    def _validate(file_path):
        if not os.path.isfile(file_path):
            pytest.fail(f"File not found: {file_path}")
        return file_path
    return _validate

# === Parameterized Test for All Parsers ===
PARSER_TESTS = [
    (parse_geometry, "geometry.txt", dict, "base"),
    (parse_material, "material.txt", dict, "base"),
    (parse_solver, "solver.txt", dict, "base"),
    (parse_boundary_conditions, "geometry.txt", np.ndarray, "base"),
    (parse_load, "load.txt", np.ndarray, "job_dirs"),
    (parse_mesh, "mesh.txt", dict, "job_dirs"),
]

@pytest.mark.parametrize("parser_func, filename, expected_type, directory", PARSER_TESTS)
def test_parsers(parser_func, filename, expected_type, directory, validate_file, base_dir_fixture, job_dirs_fixture, caplog):
    """Generic parser test function for all parser functions."""
    try:
        file_paths = []
        
        if directory == "base":
            file_path = os.path.join(base_dir_fixture, filename)
            file_paths.append(validate_file(file_path))
        elif directory == "job_dirs":
            for job_dir in job_dirs_fixture:
                file_path = os.path.join(job_dir, filename)
                if os.path.isfile(file_path):
                    file_paths.append(validate_file(file_path))
            if not file_paths:
                pytest.skip(f"File '{filename}' not found in any Job_* directories.")

        for path in file_paths:
            logger.info(f"Testing parser: {parser_func.__name__} with file: {path}")

            if parser_func == parse_mesh:
                parsed_data = parser_func(path)  # ✅ Remove geometry file dependency
            else:
                parsed_data = parser_func(path)

            debug_print(parsed_data, name=f"{parser_func.__name__} Output")
            assert isinstance(parsed_data, expected_type), f"Expected {expected_type}, but got {type(parsed_data)}"

            logger.info(f"✅ {parser_func.__name__} successfully parsed '{path}'")
    except Exception as e:
        logger.error(f"❌ Error in {parser_func.__name__}: {e}")
        pytest.fail(f"Error in {parser_func.__name__}: {e}")

# === Consistency Test: Direct Function vs ParserBase ===
@pytest.mark.parametrize("key, filename, directory", [
    ("geometry", "geometry.txt", "base"),
    ("material", "material.txt", "base"),
    ("solver", "solver.txt", "base"),
    ("load", "load.txt", "job_dirs"),
    ("boundary_conditions", "geometry.txt", "base"),
])
def test_parser_base_consistency(key, filename, directory, parser, validate_file, base_dir_fixture, job_dirs_fixture, caplog):
    """Compare individual parser output with ParserBase."""
    try:
        file_paths = []
        
        if directory == "base":
            file_path = os.path.join(base_dir_fixture, filename)
            file_paths.append(validate_file(file_path))
        elif directory == "job_dirs":
            for job_dir in job_dirs_fixture:
                file_path = os.path.join(job_dir, filename)
                if os.path.isfile(file_path):
                    file_paths.append(validate_file(file_path))
            if not file_paths:
                pytest.skip(f"File '{filename}' not found in any Job_* directories.")

        for path in file_paths:
            parser_method = getattr(parser, f"{key}_parser", None)
            direct_func = globals().get(f"parse_{key}")
            if not parser_method or not direct_func:
                pytest.fail(f"Parser method for '{key}' not found.")

            logger.info(f"Comparing {direct_func.__name__} with ParserBase for {path}")
            base_output = parser_method(path)
            direct_output = direct_func(path)
            debug_print(base_output, name=f"ParserBase {key} Output")
            debug_print(direct_output, name=f"Direct Function {key} Output")

            # Fix: Handle NumPy arrays separately
            if isinstance(base_output, np.ndarray) and isinstance(direct_output, np.ndarray):
                np.testing.assert_array_equal(base_output, direct_output)
            else:
                assert base_output == direct_output, f"Mismatch in {key} parser outputs."

            logger.info(f"✅ {key}: Direct function and ParserBase output match for {path}")
    except Exception as e:
        logger.error(f"❌ Error in consistency check for '{key}': {e}")
        pytest.fail(f"Error in consistency check for '{key}': {e}")
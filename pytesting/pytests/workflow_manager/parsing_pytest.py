# pytesting\pytests\workflow_manager\parsing_pytest.py

"""
Centralized Tests for FEM Input Parsers using `pytest`.

Run these tests from the **project root directory**:
    python -m pytest pytesting/pytests/workflow_manager_pytests/parsing_pytest.py -v --log-cli-level=INFO

All logs will be stored in `pytesting/logs/pytest_parsing.log`.
"""

import pytest
import os
import sys
import pprint
import numpy as np
import glob
import logging
import traceback
import datetime
import importlib

# === Setup Logging System ===
def find_project_root():
    """Finds the project root by looking for a marker file or falling back to PYTHONPATH."""
    current_path = os.path.abspath(__file__)
    while True:
        parent_path = os.path.dirname(current_path)
        if os.path.exists(os.path.join(parent_path, ".project_root")):
            return parent_path
        if parent_path == current_path:  # Reached the root directory
            break
        current_path = parent_path
    return os.environ.get("PYTHONPATH", os.getcwd()).split(os.pathsep)[0]

PROJECT_ROOT = find_project_root()
sys.path.insert(0, PROJECT_ROOT)
logger = logging.getLogger("pytest_parsing")
logger.setLevel(logging.INFO)
logger.info(f"Using project root: {PROJECT_ROOT}")

# Ensure log directory exists robustly
try:
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
except Exception as e:
    LOG_DIR = os.getcwd()  # Fallback to current directory
    logger.warning(f"⚠️ Failed to create log directory at {LOG_DIR}, using current directory. Error: {e}")

LOG_FILENAME = f"pytest_parsing_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE = os.path.join(LOG_DIR, LOG_FILENAME)

try:
    # File Handler (logs to file)
    file_handler = logging.FileHandler(LOG_FILE, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
except Exception as e:
    logger.warning(f"⚠️ Failed to create log file {LOG_FILE}, logging to console only. Error: {e}")

# Console Handler (logs to console)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

logger.info(f"🔥 LOGGING STARTED: Logs will be written to {LOG_FILE}")

# === Import Parsers ===
try:
    from workflow_manager.parsing.parser_base import ParserBase
    from workflow_manager.parsing.geometry_parser import parse_geometry
    from workflow_manager.parsing.material_parser import parse_material
    from workflow_manager.parsing.mesh_parser import parse_mesh
    from workflow_manager.parsing.load_parser import parse_load
    from workflow_manager.parsing.solver_parser import parse_solver
except ImportError as e:
    logger.error("❌ Failed to import parsers. Ensure workflow_manager is in PYTHONPATH.")
    raise e

# === Helper Functions ===
def format_numpy_array(array, name="Array"):
    """Formats NumPy arrays with column indices for better logging."""
    if not isinstance(array, np.ndarray):
        return str(array)
    header = "Indices: " + " ".join([f"[{i:^6}]" for i in range(min(array.shape[1], 10))])
    data_str = np.array2string(array, precision=3, threshold=10)
    return f"\n--- {name} ---\n{header}\n{data_str}"

# === Fixtures ===
@pytest.fixture(scope="session")
def base_dir_fixture():
    """Provides the base directory path for test files."""
    base_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "jobs", "base"))
    if not os.path.exists(base_dir):
        pytest.skip(f"⚠️ Base directory not found: {base_dir}")
    return base_dir

@pytest.fixture(scope="session")
def job_dirs_fixture():
    """Finds all job directories dynamically or falls back to the current directory."""
    jobs_dir = os.path.join(PROJECT_ROOT, "jobs")
    job_dirs = glob.glob(os.path.join(jobs_dir, "job_*"))
    if not job_dirs:
        pytest.skip(f"⚠️ No job_* directories found in {jobs_dir}. Falling back to current directory.")
        job_dirs = [os.getcwd()]
    return job_dirs

@pytest.fixture
def validate_file():
    def _validate(file_path):
        if not os.path.isfile(file_path):
            job_dir = os.path.dirname(file_path)
            available_files = os.listdir(job_dir) if os.path.exists(job_dir) else []
            logger.error(f"❌ File not found: {file_path}\n📂 Available files in {job_dir}: {available_files}")
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path
    return _validate

# === Test Cases ===
@pytest.mark.parametrize("parser_func, filename, expected_type, directory", [
    (parse_geometry, "geometry.txt", np.ndarray, "base"),
    (parse_material, "material.txt", np.ndarray, "base"),
    (parse_solver, "solver.txt", dict, "base"),
    (parse_load, "load.txt", np.ndarray, "job_dirs"),
    (parse_mesh, "mesh.txt", dict, "job_dirs"),
])
def test_parsers(parser_func, filename, expected_type, directory, validate_file, job_dirs_fixture):
    try:
        file_paths = [validate_file(os.path.join(job_dir, filename)) for job_dir in job_dirs_fixture]
        for path in file_paths:
            logger.info(f"Testing parser: {parser_func.__name__} with file: {path}")
            parsed_data = parser_func(path)
            logger.info(format_numpy_array(parsed_data, name=f"{parser_func.__name__} Output"))
            assert isinstance(parsed_data, expected_type), f"Expected {expected_type}, but got {type(parsed_data)}"
    except Exception as e:
        logger.error(f"❌ Error in {parser_func.__name__}: {e}\n{traceback.format_exc()}")
        pytest.fail(f"Error in {parser_func.__name__}: {e}")

@pytest.mark.parametrize("key, filename, directory", [
    ("geometry", "geometry.txt", "base"),
    ("material", "material.txt", "base"),
    ("solver", "solver.txt", "base"),
    ("load", "load.txt", "job_dirs"),
])
def test_parser_base_consistency(key, filename, directory, parser, validate_file, job_dirs_fixture):
    try:
        file_paths = [validate_file(os.path.join(job_dir, filename)) for job_dir in job_dirs_fixture]
        for path in file_paths:
            parser_method = getattr(parser, f"{key}_parser", None)
            direct_func = globals().get(f"parse_{key}")
            if not parser_method or not direct_func:
                pytest.skip(f"No parser method found for '{key}'.")
            base_output = parser_method(path)
            direct_output = direct_func(path)
            logger.info(format_numpy_array(base_output, name=f"ParserBase {key} Output"))
            logger.info(format_numpy_array(direct_output, name=f"Direct Function {key} Output"))
            assert np.array_equal(base_output, direct_output), f"Mismatch in {key} parser outputs."
    except Exception as e:
        logger.error(f"❌ Error in consistency check for '{key}': {e}\n{traceback.format_exc()}")
        pytest.fail(f"Error in consistency check for '{key}': {e}")
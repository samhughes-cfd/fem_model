"""
Automated Tests for FEM Input Parsers using `pytest`.

Run this test from the **project root directory**:

    pytest testing/TEST_PARSING.py -v

All logs will be stored in `testing/pytest_parsing.log`.
"""

import pytest
import os
import logging
import sys

# Ensure the project root is added to `sys.path` to resolve imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Path of this script
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # Move up one level

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Check if `workflow_manager` exists and has `__init__.py`
if not os.path.exists(os.path.join(PROJECT_ROOT, "workflow_manager", "__init__.py")):
    raise ImportError("Error: `workflow_manager` is missing `__init__.py`!")

# Configure logging to store logs in the same directory as TEST_PARSING.py
LOG_FILE = os.path.join(SCRIPT_DIR, "pytest_parsing.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()]
)

# Import parsers from `workflow_manager`
from workflow_manager.parsing.parser_base import ParserBase
from workflow_manager.parsing.geometry_parser import parse_geometry
from workflow_manager.parsing.material_parser import parse_material
from workflow_manager.parsing.mesh_parser import parse_mesh
from workflow_manager.parsing.load_parser import parse_load
from workflow_manager.parsing.solver_parser import parse_solver
from workflow_manager.parsing.boundary_condition_parser import parse_boundary_conditions

# Define test file paths
JOBS_DIR = os.path.join(PROJECT_ROOT, "jobs")
BASE_DIR = os.path.join(JOBS_DIR, "base")
JOB_DIR = os.path.join(JOBS_DIR, "job_0001")

@pytest.fixture
def parser():
    """Provides a ParserBase instance for all tests."""
    return ParserBase()

@pytest.fixture
def validate_file():
    """Check if a file exists before parsing."""
    def _validate(file_path):
        if not os.path.isfile(file_path):
            pytest.fail(f"File not found: {file_path}")
        return file_path
    return _validate

@pytest.mark.parametrize("parser_func, file_path", [
    (parse_geometry, os.path.join(BASE_DIR, "geometry.txt")),
    (parse_material, os.path.join(BASE_DIR, "material.txt")),
    (parse_solver, os.path.join(BASE_DIR, "solver.txt")),
    (parse_load, os.path.join(JOB_DIR, "load.txt")),
    (parse_boundary_conditions, os.path.join(BASE_DIR, "boundary_conditions.txt")),
])
def test_individual_parsers(parser_func, file_path, validate_file):
    """Test individual parsing functions and log results."""
    file_path = validate_file(file_path)
    parsed_data = parser_func(file_path)

    assert parsed_data is not None, f"Parsing failed for {file_path}"
    assert isinstance(parsed_data, dict), f"Parsed data for {file_path} should be a dictionary"

    logging.info(f"{parser_func.__name__} successfully parsed {file_path}")

def test_mesh_parser(validate_file):
    """Test the mesh parser and log results."""
    mesh_file = validate_file(os.path.join(JOB_DIR, "k_2_node_101.txt"))
    geometry_file = validate_file(os.path.join(BASE_DIR, "geometry.txt"))

    parsed_mesh = parse_mesh(mesh_file, geometry_file)
    assert parsed_mesh is not None, "Mesh parsing failed"
    assert "connectivity" in parsed_mesh, "Mesh data missing 'connectivity' key"
    assert "node_positions" in parsed_mesh, "Mesh data missing 'node_positions' key"

    logging.info("Mesh parser successfully parsed the mesh file.")

def test_parser_base(parser, validate_file):
    """Test `ParserBase` class and log results."""
    files = {
        "geometry": os.path.join(BASE_DIR, "geometry.txt"),
        "material": os.path.join(BASE_DIR, "material.txt"),
        "solver": os.path.join(BASE_DIR, "solver.txt"),
        "load": os.path.join(JOB_DIR, "load.txt"),
        "boundary_conditions": os.path.join(BASE_DIR, "boundary_conditions.txt"),
    }

    for key, path in files.items():
        file_path = validate_file(path)
        parse_method = getattr(parser, f"{key}_parser")
        parsed_data = parse_method(file_path)

        assert parsed_data is not None, f"{key} parsing failed"
        assert isinstance(parsed_data, dict), f"{key} parsed data should be a dictionary"

        logging.info(f"{key} parser in `ParserBase` successfully parsed {file_path}")

def test_parser_base_mesh(parser, validate_file):
    """Test mesh parsing through `ParserBase` and log results."""
    mesh_file = validate_file(os.path.join(JOB_DIR, "k_2_node_101.txt"))
    geometry_file = validate_file(os.path.join(BASE_DIR, "geometry.txt"))

    parsed_mesh = parser.mesh_parser(mesh_file, geometry_file)
    assert parsed_mesh is not None, "Mesh parsing failed through `ParserBase`"
    assert "connectivity" in parsed_mesh, "Mesh data missing 'connectivity' key"
    assert "node_positions" in parsed_mesh, "Mesh data missing 'node_positions' key"

    logging.info("`ParserBase` successfully parsed the mesh file.")
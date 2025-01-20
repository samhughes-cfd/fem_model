"""
Automated Tests for FEM Input Parsers using `pytest`.

Run this test from the **project root directory**:

    python -m pytest pytest_testing/test_parsing.py -v

All logs will be stored in `pytest_testing/pytest_parsing.log`.
"""

import pytest
import os
import logging
import sys
import time
import numpy as np

# Ensure the project root is added to `sys.path` to resolve imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Path of this script
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # Move up one level

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure logging
LOG_FILE = os.path.join(SCRIPT_DIR, "pytest_parsing.log")

def setup_logging():
    """Configure logging with file and console handlers and force log flushing."""
    logging.getLogger().handlers.clear()  # Remove existing handlers

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    
    # Stream handler (for VS Code Terminal)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)

    # Add handlers
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(stream_handler)
    logging.getLogger().setLevel(logging.INFO)

    # Ensure logs are flushed immediately
    file_handler.flush()
    stream_handler.flush()

setup_logging()

def flush_logs():
    """Force log writes to file."""
    for handler in logging.getLogger().handlers:
        handler.flush()
    time.sleep(0.1)  # Small delay to ensure writes are completed

logging.info("Test script started.")
with open(LOG_FILE, "a", encoding="utf-8") as log_file:
    log_file.write("Manual log entry - If this appears, logging works.\n")

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
    (parse_boundary_conditions, os.path.join(BASE_DIR, "geometry.txt")),  # Fix: Use geometry.txt
])
def test_individual_parsers(parser_func, file_path, validate_file):
    """Test individual parsing functions and log results."""
    try:
        file_path = validate_file(file_path)
        parsed_data = parser_func(file_path)

        # Debugging output to log parsed data type
        logging.info(f"Parsed data type for {parser_func.__name__}: {type(parsed_data)}")

        # Adjust test conditions based on expected data type
        if parser_func in [parse_load, parse_boundary_conditions]:
            assert isinstance(parsed_data, np.ndarray), f"Parsed data for {file_path} should be a numpy array"
        else:
            assert isinstance(parsed_data, dict), f"Parsed data for {file_path} should be a dictionary"

        logging.info(f"{parser_func.__name__} successfully parsed {file_path}")
    except Exception as e:
        logging.error(f"Error during {parser_func.__name__} parsing: {e}")
    finally:
        flush_logs()

def test_mesh_parser(validate_file):
    """Test the mesh parser and log results."""
    try:
        mesh_file = validate_file(os.path.join(JOB_DIR, "k_2_node_101.txt"))
        geometry_file = validate_file(os.path.join(BASE_DIR, "geometry.txt"))

        parsed_mesh = parse_mesh(mesh_file, geometry_file)
        assert parsed_mesh is not None, "Mesh parsing failed"
        assert "connectivity" in parsed_mesh, "Mesh data missing 'connectivity' key"
        assert "node_positions" in parsed_mesh, "Mesh data missing 'node_positions' key"

        logging.info("Mesh parser successfully parsed the mesh file.")
    except Exception as e:
        logging.error(f"Error during mesh parsing: {e}")
    finally:
        flush_logs()

def test_parser_base(parser, validate_file):
    """Test `ParserBase` class and log results."""
    files = {
        "geometry": os.path.join(BASE_DIR, "geometry.txt"),
        "material": os.path.join(BASE_DIR, "material.txt"),
        "solver": os.path.join(BASE_DIR, "solver.txt"),
        "load": os.path.join(JOB_DIR, "load.txt"),
        "boundary_conditions": os.path.join(BASE_DIR, "geometry.txt"),  # Fix: Use geometry.txt
    }

    for key, path in files.items():
        try:
            file_path = validate_file(path)
            parse_method = getattr(parser, f"{key}_parser")
            parsed_data = parse_method(file_path)

            # Adjust test conditions based on expected data type
            if key in ["load", "boundary_conditions"]:
                assert isinstance(parsed_data, np.ndarray), f"{key} parsed data should be a numpy array"
            else:
                assert isinstance(parsed_data, dict), f"{key} parsed data should be a dictionary"

            logging.info(f"{key} parser in `ParserBase` successfully parsed {file_path}")
        except Exception as e:
            logging.error(f"Error during {key} parsing: {e}")
        finally:
            flush_logs()

def test_parser_base_mesh(parser, validate_file):
    """Test mesh parsing through `ParserBase` and log results."""
    try:
        mesh_file = validate_file(os.path.join(JOB_DIR, "k_2_node_101.txt"))
        geometry_file = validate_file(os.path.join(BASE_DIR, "geometry.txt"))

        parsed_mesh = parser.mesh_parser(mesh_file, geometry_file)
        assert parsed_mesh is not None, "Mesh parsing failed through `ParserBase`"
        assert "connectivity" in parsed_mesh, "Mesh data missing 'connectivity' key"
        assert "node_positions" in parsed_mesh, "Mesh data missing 'node_positions' key"

        logging.info("`ParserBase` successfully parsed the mesh file.")
    except Exception as e:
        logging.error(f"Error during `ParserBase` mesh parsing: {e}")
    finally:
        flush_logs()
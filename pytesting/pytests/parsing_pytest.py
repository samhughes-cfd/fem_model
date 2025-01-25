# pytesting/pytests/parsing_pytest.py

"""
Comprehensive Tests for FEM Input Parsers using `pytest`.

Run tests:
    python -m pytest pytesting/pytests/parsing_pytest.py -v --log-cli-level=INFO

Logs stored in:
    pytesting/logs/parsing_pytest.log
"""

import pytest
import numpy as np
import os
import logging
import pprint
from pre_processing.parsing.geometry_parser import parse_geometry
from pre_processing.parsing.material_parser import parse_material
from pre_processing.parsing.mesh_parser import parse_mesh
from pre_processing.parsing.load_parser import parse_load
from pre_processing.parsing.solver_parser import parse_solver
from pre_processing.parsing.base_parser import ParserBase   

# Ensure the logs directory exists
log_dir = os.path.join("pytesting", "logs")
os.makedirs(log_dir, exist_ok=True)

# Log file path
log_file = os.path.join(log_dir, "test_parsing.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler()
    ]
)

pp = pprint.PrettyPrinter(indent=2, width=80)  # Pretty print for better structure

logging.info("📜 Logging initialized: Writing logs to pytesting/logs/test_parsing.log")

@pytest.mark.parametrize("parser_func, expected_shape", [
    (parse_geometry, (1, 20)),
    (parse_material, (1, 4)),
])
def test_parsers_empty_file(parser_func, expected_shape, tmp_path):
    """Test parsers with an empty file and ensure they return NaN-filled arrays."""
    
    test_file = tmp_path / "empty.txt"
    test_file.write_text("")

    logging.info(f"Testing empty file parsing with {parser_func.__name__}...")
    
    output = parser_func(str(test_file))
    
    # Print and log output
    output_str = f"\n📊 Parsed Data from {parser_func.__name__}:\n{pp.pformat(output)}"
    print(output_str)
    logging.info(output_str)

    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    assert np.all(np.isnan(output)), "Expected all values to be NaN"


@pytest.mark.parametrize("parser_func, filename, directory, expected_columns", [
    (parse_load, "load.txt", "job_0001", 9),
])
def test_parsers(parser_func, filename, directory, expected_columns, tmp_path):
    """Test parsers independently of `loads.txt` content."""

    base_path = tmp_path / "jobs" / directory
    base_path.mkdir(parents=True, exist_ok=True)

    file_path = base_path / filename
    file_path.touch()  # Ensure the file exists, even if empty

    logging.info(f"Testing parser {parser_func.__name__} with {file_path}...")

    output = parser_func(str(file_path))

    # Print and log output
    output_str = f"\n📊 Parsed Data from {parser_func.__name__}:\n{pp.pformat(output)}"
    print(output_str)
    logging.info(output_str)

    # Check that the output has the expected number of columns
    assert output.shape[1] == expected_columns, f"Expected {expected_columns} columns, got {output.shape[1]}"

    # If the file was empty, output should also be empty
    if file_path.stat().st_size == 0:
        assert output.shape[0] == 0, "Expected empty array for empty file"


@pytest.mark.parametrize("parser_func, parser_base_method, filename", [
    (parse_geometry, "geometry_parser", "geometry.txt"),
    (parse_material, "material_parser", "material.txt"),
    (parse_load, "load_parser", "load.txt"),
    (parse_mesh, "mesh_parser", "mesh.txt"),
    (parse_solver, "solver_parser", "solver.txt"),
])
def test_parser_base_consistency(parser_func, parser_base_method, filename, tmp_path):
    """Ensure individual parsers return the same output as ParserBase."""

    base_path = tmp_path / "jobs" / "base"
    base_path.mkdir(parents=True, exist_ok=True)

    file_path = base_path / filename
    file_path.touch()  # Ensure the file exists

    assert os.path.isfile(file_path), f"File not found: {file_path}"

    logging.info(f"Testing base consistency for {parser_func.__name__} using {file_path}...")

    parser_base = ParserBase()

    # ✅ Call individual parser
    parser_output = parser_func(str(file_path))

    # ✅ Call ParserBase method dynamically
    parser_base_method_func = getattr(parser_base, parser_base_method)  # Get method dynamically
    base_output = parser_base_method_func(str(file_path))

    # Print and log outputs
    parser_output_str = f"\n📊 Output from {parser_func.__name__}:\n{pp.pformat(parser_output)}"
    base_output_str = f"\n📊 Output from ParserBase.{parser_base_method}:\n{pp.pformat(base_output)}"
    
    print(parser_output_str)
    print(base_output_str)
    
    logging.info(parser_output_str)
    logging.info(base_output_str)

    # ✅ Compare outputs
    assert isinstance(base_output, type(parser_output)), f"Inconsistent output type for {parser_func.__name__}"
    if isinstance(base_output, np.ndarray):
        assert np.array_equal(base_output, parser_output), f"Output mismatch for {parser_func.__name__}"
    elif isinstance(base_output, tuple):
        assert all(np.array_equal(b, p) for b, p in zip(base_output, parser_output)), f"Tuple output mismatch for {parser_func.__name__}"

    logging.info(f"✅ {parser_func.__name__} produces consistent output with ParserBase.")
# pytesting/pytests/pytest_element_parent_class.py

"""
Centralized Tests for the FEM Parent Element Class (`Element1DBase`) using `pytest`.

Run these tests from the **project root directory**:
    python -m pytest pre_processing/pytesting/pytests/pytest_element_parent_class.py -v --log-cli-level=INFO

All logs will be stored in `pre_processing/pytesting/logs/pytest_element_parent_class.log`.
"""

import pytest
import os
import numpy as np
import logging

# === Setup Logging System ===
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../logs")
LOG_FILE = os.path.join(LOG_DIR, "pytest_element_parent_class.log")
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure log directory exists

logger = logging.getLogger("pytest_element_parent_class")
logger.setLevel(logging.INFO)

# Log to file (but avoid console duplication)
file_handler = logging.FileHandler(LOG_FILE, mode="w")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

logger.info("🔥 LOGGING STARTED: Logs will be written to %s", LOG_FILE)

# === Import Parsers & Element Base(s) ===
from pre_processing.element_library.element_1D_base import Element1DBase
from pre_processing.parsing.material_parser import parse_material
from pre_processing.parsing.geometry_parser import parse_geometry
from pre_processing.parsing.mesh_parser import parse_mesh
from pre_processing.parsing.load_parser import parse_load

# === Load Parsed Data from Files ===
@pytest.fixture(scope="module")
def parsed_data():
    """Loads the pre-parsed FEM data for element testing."""
    base_dir = os.getenv("FEM_BASE_DIR", "../../../jobs/base")
    job_dir = os.getenv("FEM_JOB_DIR", "../../../jobs/job_0001")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), base_dir))
    job_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), job_dir))

    material_props = parse_material(os.path.join(base_dir, "material.txt"))
    geometry_data = parse_geometry(os.path.join(base_dir, "geometry.txt"))
    mesh_data = parse_mesh(os.path.join(job_dir, "mesh.txt"))
    loads_array = parse_load(os.path.join(job_dir, "load.txt"))

    return material_props, geometry_data, mesh_data, loads_array

# === Tests Using Parsed Data ===
@pytest.mark.parametrize("element_id", range(5))
def test_get_element_connectivity(parsed_data, element_id, caplog):
    """Tests retrieval of connected nodes from parsed mesh data."""
    with caplog.at_level(logging.INFO):
        material_props, geometry_data, mesh_data, loads_array = parsed_data
        element = Element1DBase(
            element_id, material_props, geometry_data, mesh_data, mesh_data["node_positions"], loads_array
        )

        expected_connectivity = mesh_data["connectivity"][element_id]
        assert element.get_element_connectivity() == expected_connectivity, f"Connectivity mismatch for element {element_id}!"


@pytest.mark.parametrize("element_id", range(5))
def test_get_node_coordinates(parsed_data, element_id):
    """Tests retrieval of node coordinates from parsed mesh data."""
    material_props, geometry_data, mesh_data, loads_array = parsed_data
    element = Element1DBase(
        element_id, material_props, geometry_data, mesh_data, mesh_data["node_positions"], loads_array
    )

    node_ids = mesh_data["connectivity"][element_id]
    expected_coords = np.array([mesh_data["node_positions"][node_id - 1] for node_id in node_ids])
    np.testing.assert_array_equal(
        element.get_node_coordinates(), expected_coords,
        err_msg=f"Node coordinates mismatch for element {element_id}!"
    )


@pytest.mark.parametrize("element_id", range(5))
def test_get_element_length(parsed_data, element_id):
    """Tests retrieval of element length from parsed mesh data."""
    material_props, geometry_data, mesh_data, loads_array = parsed_data
    element = Element1DBase(
        element_id, material_props, geometry_data, mesh_data, mesh_data["node_positions"], loads_array
    )

    element_length = element.get_element_length()
    assert isinstance(element_length, float), f"Expected float but got {type(element_length)} with value {element_length}"
    assert element_length == float(mesh_data["element_lengths"][element_id]), f"Incorrect element length for element {element_id}!"


@pytest.mark.parametrize("element_id", range(5))
def test_get_element_loads(parsed_data, element_id):
    """Tests retrieval of element loads from parsed loads array."""
    material_props, geometry_data, mesh_data, loads_array = parsed_data
    element = Element1DBase(
        element_id, material_props, geometry_data, mesh_data, mesh_data["node_positions"], loads_array
    )

    node_ids = mesh_data["connectivity"][element_id]
    expected_loads = loads_array[[node_ids[0] - 1, node_ids[1] - 1], :]  # Shape: (2,6)
    np.testing.assert_array_equal(
        element.get_element_loads(), expected_loads,
        err_msg=f"Load mismatch for element {element_id}!"
    )


# === Additional Edge Case Tests ===
def test_missing_mesh_file(monkeypatch):
    """Tests behavior when mesh file is missing."""
    monkeypatch.setattr("workflow_manager.parsing.mesh_parser.parse_mesh", lambda _: None)

    with pytest.raises(TypeError, match="object is not subscriptable"):
        parsed_data()


@pytest.mark.parametrize("element_id", [-1, 100])  # Negative and out-of-bounds index
def test_invalid_element_id(parsed_data, element_id):
    """Tests behavior for invalid element IDs."""
    material_props, geometry_data, mesh_data, loads_array = parsed_data

    with pytest.raises(IndexError, match="list index out of range"):
        Element1DBase(
            element_id, material_props, geometry_data, mesh_data, mesh_data["node_positions"], loads_array
        )
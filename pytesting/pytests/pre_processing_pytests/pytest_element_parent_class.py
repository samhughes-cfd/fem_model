# pre_processing\pytesting\pytests\pytest_element_parent_class.py

"""
Centralized Tests for the FEM Parent Element Class (`Element1DBase`) using `pytest`.

Run these tests from the **project root directory**:
    python -m pytest pre_processing/pytesting/pytests/pytest_element_parent_class.py -v --log-cli-level=INFO

All logs will be stored in `pre_processing/pytesting/logs/pytest_element_parent_class.log`.
"""


import pytest
import os
import pprint
import numpy as np
import glob
import logging

# === Setup Logging System ===
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../logs")
LOG_FILE = os.path.join(LOG_DIR, "pytest_element_parent_class.log")

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Create logger
logger = logging.getLogger("pytest_element_parent_class")
logger.setLevel(logging.INFO)

# File Handler (logs to file)
file_handler = logging.FileHandler(LOG_FILE, mode="w")  # Overwrites log each time
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Console Handler (logs to console)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Attach handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"🔥 LOGGING STARTED: Logs will be written to {LOG_FILE}")

# === Import Parsers & Element Base(s) ===
from pre_processing.element_library.element_1D_base import Element1DBase
from workflow_manager.parsing.material_parser import parse_material
from workflow_manager.parsing.geometry_parser import parse_geometry
from workflow_manager.parsing.mesh_parser import parse_mesh
from workflow_manager.parsing.load_parser import parse_load

# === Load Parsed Data from Files ===
@pytest.fixture(scope="module")
def parsed_data():
    """Loads the pre-parsed FEM data for element testing."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../jobs/base"))
    job_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../jobs/job_0001"))

    material_props = parse_material(os.path.join(base_dir, "material.txt"))
    geometry_data = parse_geometry(os.path.join(base_dir, "geometry.txt"))
    mesh_data = parse_mesh(os.path.join(job_dir, "mesh.txt"))
    loads_array = parse_load(os.path.join(job_dir, "load.txt"))

    return material_props, geometry_data, mesh_data, loads_array

# === Tests Using Parsed Data ===
@pytest.mark.parametrize("element_id", range(5))  # Test the first 5 elements
def test_get_element_connectivity(parsed_data, element_id):
    """Tests retrieval of connected nodes from parsed mesh data."""
    material_props, geometry_data, mesh_data, loads_array = parsed_data
    element = Element1DBase(
        element_id,
        material_props,
        geometry_data,
        mesh_data,
        mesh_data["node_positions"],
        loads_array
    )

    expected_connectivity = mesh_data["connectivity"][element_id]
    assert element.get_element_connectivity() == expected_connectivity, "Connectivity mismatch!"

@pytest.mark.parametrize("element_id", range(5))
def test_get_node_coordinates(parsed_data, element_id):
    """Tests retrieval of node coordinates from parsed mesh data."""
    material_props, geometry_data, mesh_data, loads_array = parsed_data
    element = Element1DBase(
        element_id,
        material_props,
        geometry_data,
        mesh_data,
        mesh_data["node_positions"],
        loads_array
    )

    node_ids = mesh_data["connectivity"][element_id]
    expected_coords = np.array([mesh_data["node_positions"][node_id - 1] for node_id in node_ids])
    np.testing.assert_array_equal(element.get_node_coordinates(), expected_coords)

@pytest.mark.parametrize("element_id", range(5))
def test_get_element_length(parsed_data, element_id):
    """Tests retrieval of element length from parsed mesh data."""
    material_props, geometry_data, mesh_data, loads_array = parsed_data
    element = Element1DBase(
        element_id,
        material_props,
        geometry_data,
        mesh_data,
        mesh_data["node_positions"],
        loads_array
    )

    assert isinstance(element.get_element_length(), float), "Element length is not a float!"
    assert element.get_element_length() == float(mesh_data["element_lengths"][element_id]), "Incorrect element length!"

@pytest.mark.parametrize("element_id", range(5))
def test_get_element_loads(parsed_data, element_id):
    """Tests retrieval of element loads from parsed loads array."""
    material_props, geometry_data, mesh_data, loads_array = parsed_data
    element = Element1DBase(
        element_id,
        material_props,
        geometry_data,
        mesh_data,
        mesh_data["node_positions"],
        loads_array
    )

    node_ids = mesh_data["connectivity"][element_id]
    expected_loads = loads_array[[node_ids[0] - 1, node_ids[1] - 1], :]  # Shape: (2,6)
    np.testing.assert_array_equal(element.get_element_loads(), expected_loads)

import pytest
import numpy as np
from pre_processing.element_library.element_1D_base import Element1DBase
from workflow_manager.parsing.material_parser import parse_material
from workflow_manager.parsing.geometry_parser import parse_geometry
from workflow_manager.parsing.mesh_parser import parse_mesh
from workflow_manager.parsing.load_parser import parse_load

# === Load Parsed Data from Files ===
@pytest.fixture(scope="module")
def parsed_data():
    """Loads the pre-parsed FEM data for element testing."""
    base_dir = "C:/Users/s1834431/Code/fem_model/jobs/base"
    job_dir = "C:/Users/s1834431/Code/fem_model/jobs/job_0001"

    material_props = parse_material(f"{base_dir}/material.txt")
    geometry_data = parse_geometry(f"{base_dir}/geometry.txt")
    mesh_data = parse_mesh(f"{job_dir}/mesh.txt")
    loads_array = parse_load(f"{job_dir}/load.txt")

    return material_props, geometry_data, mesh_data, loads_array

# === Tests Using Parsed Data ===

@pytest.mark.parametrize("element_id", range(5))  # Test the first 5 elements
def test_get_element_connectivity(parsed_data, element_id):
    """Tests retrieval of connected nodes from parsed mesh data."""
    _, _, mesh_data, _ = parsed_data
    element = Element1DBase(element_id, *parsed_data)

    assert element.get_element_connectivity() == mesh_data["connectivity"][element_id], "Connectivity mismatch!"

@pytest.mark.parametrize("element_id", range(5))
def test_get_node_coordinates(parsed_data, element_id):
    """Tests retrieval of node coordinates from parsed mesh data."""
    _, _, mesh_data, _ = parsed_data
    element = Element1DBase(element_id, *parsed_data)

    node_ids = mesh_data["connectivity"][element_id]
    expected_coords = np.array([mesh_data["node_positions"][node_id - 1] for node_id in node_ids])
    np.testing.assert_array_equal(element.get_node_coordinates(), expected_coords)

@pytest.mark.parametrize("element_id", range(5))
def test_get_element_length(parsed_data, element_id):
    """Tests retrieval of element length from parsed mesh data."""
    _, _, mesh_data, _ = parsed_data
    element = Element1DBase(element_id, *parsed_data)

    assert isinstance(element.get_element_length(), float), "Element length is not a float!"
    assert element.get_element_length() == float(mesh_data["element_lengths"][element_id]), "Incorrect element length!"

@pytest.mark.parametrize("element_id", range(5))
def test_get_element_loads(parsed_data, element_id):
    """Tests retrieval of element loads from parsed loads array."""
    _, _, mesh_data, loads_array = parsed_data
    element = Element1DBase(element_id, *parsed_data)

    node_ids = mesh_data["connectivity"][element_id]
    expected_loads = loads_array[[node_ids[0] - 1, node_ids[1] - 1], :]  # Shape: (2,6)
    np.testing.assert_array_equal(element.get_element_loads(), expected_loads)
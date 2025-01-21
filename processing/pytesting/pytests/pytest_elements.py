# File: testing/tests/pytest_elements.py

"""
Centralized Tests for FEM Element Stiffness (`Ke`) and Force Vectors (`Fe`).

Run these tests using:
    python -m pytest testing/pytests/pytest_elements.py -v --log-cli-level=INFO

All logs are stored in `testing/logs/pytest_elements.log` and `testing/logs/element_matrices.log`.
"""

import pytest
import os
import logging
import numpy as np
import scipy.sparse
from pathlib import Path

# === Setup Logging System ===
LOG_DIR = Path(__file__).resolve().parent / "../logs"
LOG_FILE = LOG_DIR / "pytest_elements.log"
LOG_FILE_MATRICES = LOG_DIR / "element_matrices.log"

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Main logger
logger = logging.getLogger("pytest_elements")
logger.setLevel(logging.INFO)

# File handler for detailed logs
file_handler = logging.FileHandler(LOG_FILE, mode="w")  # Overwrite logs
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Console handler for stdout
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Attach handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Matrix logger for separate storage
matrix_logger = logging.getLogger("MATRIX_LOG")
matrix_logger.setLevel(logging.INFO)
matrix_file_handler = logging.FileHandler(LOG_FILE_MATRICES, mode="w")
matrix_file_handler.setFormatter(logging.Formatter("%(message)s"))
matrix_logger.addHandler(matrix_file_handler)

logger.info(f"🔥 LOGGING STARTED: Logs will be written to {LOG_FILE}")

# === Import Element Classes ===
from pre_processing.element_library.euler_bernoulli.euler_bernoulli import EulerBernoulliBeamElement

# Element class mapping
ELEMENT_CLASSES = {
    "EulerBernoulliBeamElement": EulerBernoulliBeamElement,
}

# === Helper Functions ===
def debug_matrix(matrix, name, element_id):
    """Logs detailed matrix information."""
    matrix_logger.info(f"\nElement {element_id} {name}:\n{matrix}")

# === Fixtures ===
@pytest.fixture(scope="session")
def element_data():
    """Loads required FEM data: material, geometry, mesh, and loads."""
    from workflow_manager.parsing.material_parser import parse_material
    from workflow_manager.parsing.geometry_parser import parse_geometry
    from workflow_manager.parsing.mesh_parser import parse_mesh
    from workflow_manager.parsing.load_parser import parse_load

    base_dir = Path(__file__).resolve().parent / "../../jobs/base"
    job_dir = Path(__file__).resolve().parent / "../../jobs/job_0001"

    try:
        material_props = parse_material(base_dir / "material.txt")
        geometry_data = parse_geometry(base_dir / "geometry.txt")
        mesh_data = parse_mesh(job_dir / "mesh.txt")
        loads_array = parse_load(job_dir / "load.txt")

        logger.info("Successfully loaded element test data.")
        return material_props, geometry_data, mesh_data, loads_array
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        pytest.fail("Failed to load element test data.")

# === Centralized Parameterized Tests ===
ELEMENT_TESTS = [
    ("stiffness", "Ke", (12, 12), "element_stiffness_matrix"),
    ("force_vector", "Fe", (12,), "element_force_vector"),
]

@pytest.mark.parametrize("test_type, matrix_name, expected_shape, method_name", ELEMENT_TESTS)
@pytest.mark.parametrize("element_class", ELEMENT_CLASSES.values())
def test_element_matrices(element_class, test_type, matrix_name, expected_shape, method_name, element_data):
    """
    Generic test for element stiffness (`Ke`) and force vector (`Fe`) computation.
    Ensures correct shape, symmetry, and other properties.
    """
    material_props, geometry_data, mesh_data, loads_array = element_data

    for element_id, (node1, node2) in enumerate(mesh_data["connectivity"]):
        try:
            # Initialize element
            element = element_class(
                element_id=element_id,
                material=material_props,
                section_props=geometry_data,
                mesh_data=mesh_data,
                node_positions=mesh_data["node_positions"],
                loads_array=loads_array
            )

            # Call the computation method
            getattr(element, method_name)()
            matrix = getattr(element, matrix_name)

            # Log and validate results
            logger.info(f"{element_class.__name__} Element {element_id}: {matrix_name} computed (shape={matrix.shape})")
            debug_matrix(matrix, matrix_name, element_id)

            # Shape validation
            assert matrix.shape == expected_shape, (
                f"{element_class.__name__} Element {element_id}: {matrix_name} should have shape {expected_shape}, "
                f"but got shape {matrix.shape}"
            )

            # Additional checks
            if test_type == "stiffness":
                # Symmetry check
                assert np.allclose(matrix, matrix.T), f"{element_class.__name__} Element {element_id}: {matrix_name} is not symmetric"

                # Positive definiteness
                try:
                    np.linalg.cholesky(matrix)
                except np.linalg.LinAlgError:
                    pytest.fail(f"{element_class.__name__} Element {element_id}: {matrix_name} is not positive definite!")
            
            elif test_type == "force_vector":
                # Compare force vector with applied loads
                applied_forces = loads_array[element_id]
                assert np.allclose(matrix, applied_forces, atol=1e-6), (
                    f"{element_class.__name__} Element {element_id}: {matrix_name} does not match applied loads "
                    f"(expected {applied_forces}, got {matrix})"
                )

        except Exception as e:
            logger.error(f"Error testing {element_class.__name__} Element {element_id}: {e}")
            pytest.fail(f"Failed {matrix_name} test for {element_class.__name__} Element {element_id}")

def debug_matrix(Ke_full, Ke_reduced, Fe_full, Fe_reduced, element_id):
    """Logs full and reduced element stiffness matrices and force vectors in a structured format with DOF labels."""
    try:
        # Define DOF labels
        dof_labels_full = ["Ux1", "Uy1", "Uz1", "Rx1", "Ry1", "Rz1",
                           "Ux2", "Uy2", "Uz2", "Rx2", "Ry2", "Rz2"]
        dof_labels_reduced = ["Ux1", "Uy1", "Rz1", "Ux2", "Uy2", "Rz2"]

        # Ensure Fe vectors are in column format
        Fe_full_col = Fe_full.reshape(-1, 1)  # Shape: (12,1)
        Fe_reduced_col = Fe_reduced.reshape(-1, 1)  # Shape: (6,1)

        # Stack stiffness and force matrices side by side
        Ke_Fe_full = np.hstack((Ke_full, Fe_full_col))  # Shape: (12, 13)
        Ke_Fe_reduced = np.hstack((Ke_reduced, Fe_reduced_col))  # Shape: (6, 7)

        # Convert matrices to formatted strings with DOF labels
        def format_matrix(matrix, row_labels, col_labels, fe_label):
            """Formats a stiffness matrix with DOF labels as row and column headers."""
            formatted_str = "       " + "  ".join(col_labels) + f"  | {fe_label}\n"
            for i, row in enumerate(matrix):
                row_str = "  ".join(f"{val:8.4f}" for val in row)
                formatted_str += f"{row_labels[i]:<5}  {row_str}\n"
            return formatted_str

        # Format matrices
        Ke_Fe_full_str = format_matrix(Ke_Fe_full, dof_labels_full, dof_labels_full, "Fe_full")
        Ke_Fe_reduced_str = format_matrix(Ke_Fe_reduced, dof_labels_reduced, dof_labels_reduced, "Fe_reduced")

        # Log formatted output
        matrix_logger.info(f"\nElement {element_id}:")

        matrix_logger.info("\nKe Full:")
        matrix_logger.info(Ke_Fe_full_str)

        matrix_logger.info("\nKe Reduced:")
        matrix_logger.info(Ke_Fe_reduced_str)

    except Exception as e:
        logger.error(f"Error printing matrices for Element {element_id}: {e}")


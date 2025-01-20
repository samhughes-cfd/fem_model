# testing\tests\test_elements.py

"""
Tests individual elements in `pre_processing`, ensuring element stiffness (`Ke`) and force vectors (`Fe`) 
are correctly computed before assembly.

Run these tests using:
    python -m pytest testing/tests/pytest_elements.py -v --log-cli-level=INFO

All logs are stored in `pytest_testing/logs/pytest_elements.log` and `element_matrices.log`.
"""

import pytest
import os
import sys
import logging
import numpy as np
import scipy.sparse
from pathlib import Path

# === Setup Logging System ===

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()
LOGS_DIR = PROJECT_ROOT / "pytest_testing" / "logs"

# Ensure log directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE_MAIN = LOGS_DIR / "pytest_elements.log"
LOG_FILE_MATRICES = LOGS_DIR / "element_matrices.log"

# Create main logger
logger = logging.getLogger("pytest_elements")
logger.setLevel(logging.INFO)

# Create File Handler (logs to file)
file_handler = logging.FileHandler(LOG_FILE_MAIN, mode="w")  # Overwrites log each time
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Create Console Handler (logs to console)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Attach handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"🔥 LOGGING STARTED: Logs will be written to {LOG_FILE_MAIN}")

# Configure a separate logger for element matrices
matrix_logger = logging.getLogger("MATRIX_LOG")
matrix_logger.setLevel(logging.INFO)
matrix_file_handler = logging.FileHandler(LOG_FILE_MATRICES, mode="w")  # Overwrites log each time
matrix_file_handler.setFormatter(logging.Formatter("%(message)s"))  # Simplify matrix log format
matrix_logger.addHandler(matrix_file_handler)

# === Import FEM Components from workflow_manager ===
from workflow_manager.parsing.parser_base import ParserBase
from workflow_manager.parsing.mesh_parser import parse_mesh
from workflow_manager.parsing.geometry_parser import parse_geometry
from workflow_manager.parsing.material_parser import parse_material
from workflow_manager.parsing.load_parser import parse_load

# === Import Element Classes ===
from pre_processing.element_library.euler_bernoulli.euler_bernoulli import EulerBernoulliBeamElement
# from pre_processing.element_library.timoshenko.timoshenko import TimoshenkoBeamElement

# Element class mapping
ELEMENT_CLASSES = {
    "EulerBernoulliBeamElement": EulerBernoulliBeamElement,
    # "TimoshenkoBeamElement": TimoshenkoBeamElement,
}

# Paths for test input files
JOBS_DIR = PROJECT_ROOT / "jobs"
BASE_DIR = JOBS_DIR / "base"
JOB_DIR = JOBS_DIR / "job_0001"

# === Fixtures ===
@pytest.fixture(scope="session")
def parser():
    """Provides a ParserBase instance for parsing structured FEM input."""
    return ParserBase()

@pytest.fixture(scope="session")
def element_data(parser):
    """Loads required FEM data: material, geometry, mesh, and loads."""
    try:
        material_props = parse_material(BASE_DIR / "material.txt")
        geometry_data = parse_geometry(BASE_DIR / "geometry.txt")
        mesh_data = parse_mesh(JOB_DIR / "k_2_node_101.txt")  # ✅ Uses new `parse_mesh`
        loads_array = parse_load(JOB_DIR / "load.txt")

        logger.info("✅ Successfully loaded element test data.")
        return material_props, geometry_data, mesh_data, loads_array
    except Exception as e:
        logger.error(f"❌ Error loading test data: {e}")
        pytest.fail("Failed to load element test data.")

# === Element Stiffness Tests ===
@pytest.mark.parametrize("element_class", ELEMENT_CLASSES.values())
def test_element_stiffness(element_class, element_data):
    """Tests stiffness matrix `Ke` computation for each element type."""
    material_props, geometry_data, mesh_data, loads_array = element_data

    for element_id, (node1, node2) in enumerate(mesh_data["connectivity"]):
        try:
            element = element_class(
                element_id=element_id,
                material=material_props,
                section_props=geometry_data,
                mesh_data=mesh_data,
                node_positions=mesh_data["node_positions"],
                loads_array=loads_array
            )

            element.element_stiffness_matrix()
            Ke = element.Ke

            logger.info(f"✅ {element_class.__name__} Element {element_id}: Ke computed (shape={Ke.shape})")

            # Store Ke in separate log
            matrix_logger.info(f"\nElement {element_id} Ke:\n{Ke}")

            # Validate shape
            assert Ke.shape == (12, 12), (
                f"❌ {element_class.__name__} Element {element_id}: Ke should be (12,12), "
                f"but got shape {Ke.shape}"
            )

            # Symmetry Check
            assert np.allclose(Ke, Ke.T), f"❌ {element_class.__name__} Element {element_id}: Ke is not symmetric!"

            # Positive Definiteness Check
            try:
                np.linalg.cholesky(Ke)
            except np.linalg.LinAlgError:
                pytest.fail(f"❌ {element_class.__name__} Element {element_id}: Ke is not positive definite!")

            # Optional: Convert to sparse for large models
            Ke_sparse = scipy.sparse.csr_matrix(Ke)
            logger.info(f"🟢 {element_class.__name__} Element {element_id}: Ke converted to sparse (nnz={Ke_sparse.nnz})")

        except Exception as e:
            logger.error(f"❌ Error testing {element_class.__name__} Element {element_id}: {e}")
            pytest.fail(f"Failed Ke test for {element_class.__name__} Element {element_id}")

# === Element Force Vector Tests ===
@pytest.mark.parametrize("element_class", ELEMENT_CLASSES.values())
def test_element_force_vector(element_class, element_data):
    """Tests force vector `Fe` computation for each element type."""
    material_props, geometry_data, mesh_data, loads_array = element_data

    for element_id, (node1, node2) in enumerate(mesh_data["connectivity"]):
        try:
            element = element_class(
                element_id=element_id,
                material=material_props,
                section_props=geometry_data,
                mesh_data=mesh_data,
                node_positions=mesh_data["node_positions"],
                loads_array=loads_array
            )

            element.element_force_vector()
            Fe = element.Fe

            logger.info(f"✅ {element_class.__name__} Element {element_id}: Fe computed (shape={Fe.shape})")

            # Store Fe in separate log
            matrix_logger.info(f"\nElement {element_id} Fe:\n{Fe}")

            # Validate shape
            assert Fe.shape == (12,), (
                f"❌ {element_class.__name__} Element {element_id}: Fe should be (12,), "
                f"but got shape {Fe.shape}"
            )

            # Validate that `Fe` includes applied loads
            applied_forces = loads_array[element_id]
            assert np.allclose(Fe, applied_forces, atol=1e-6), (
                f"❌ {element_class.__name__} Element {element_id}: Fe does not match applied loads! "
                f"Expected {applied_forces}, got {Fe}"
            )

        except Exception as e:
            logger.error(f"❌ Error testing {element_class.__name__} Element {element_id}: {e}")
            pytest.fail(f"Failed Fe test for {element_class.__name__} Element {element_id}")
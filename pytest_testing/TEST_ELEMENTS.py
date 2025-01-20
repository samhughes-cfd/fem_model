"""
Tests individual elements in `pre_processing`, ensuring element stiffness (`Ke`) and force vectors (`Fe`) 
are correctly computed before assembly.
"""

import pytest
import sys
import logging
import numpy as np
import scipy.sparse
from pathlib import Path

# Add project root for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Define the logs directory relative to PROJECT_ROOT
LOGS_DIR = PROJECT_ROOT / "pytest_testing" / "logs"

# Create the logs directory if it doesn't exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Define log file paths within the logs directory
LOG_FILE_MAIN = LOGS_DIR / "pytest_elements.log"
LOG_FILE_MATRICES = LOGS_DIR / "element_matrices.log"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE_MAIN, mode="w")  # Overwrite mode; use "a" to append
    ]
)

# Configure a separate logger for matrices
matrix_logger = logging.getLogger("MATRIX_LOG")
matrix_logger.setLevel(logging.INFO)
matrix_file_handler = logging.FileHandler(LOG_FILE_MATRICES, mode="w")  # Overwrite mode; use "a" to append
matrix_file_handler.setFormatter(logging.Formatter("%(message)s"))  # Simplify matrix log format
matrix_logger.addHandler(matrix_file_handler)

# FEM Imports
from workflow_manager.parsing.parser_base import ParserBase
from pre_processing.element_library.euler_bernoulli.euler_bernoulli import EulerBernoulliBeamElement
# from pre_processing.element_library.timoshenko.timoshenko import TimoshenkoBeamElement
from pre_processing.element_library.utilities.shape_function_library.euler_bernoulli_sf import euler_bernoulli_shape_functions
from pre_processing.element_library.utilities.jacobian import compute_jacobian_matrix, compute_jacobian_determinant
from pre_processing.element_library.utilities.gauss_quadrature import integrate_matrix
from pre_processing.element_library.utilities.dof_mapping import expand_dof_mapping

# Element class mapping
ELEMENT_CLASSES = {
    "EulerBernoulliBeamElement": EulerBernoulliBeamElement,
    # "TimoshenkoBeamElement": TimoshenkoBeamElement,
}

# Paths for test input files
JOBS_DIR = PROJECT_ROOT / "jobs"
BASE_DIR = JOBS_DIR / "base"
JOB_DIR = JOBS_DIR / "job_0001"

@pytest.fixture
def parser():
    """Provides a ParserBase instance for parsing structured FEM input."""
    return ParserBase()

@pytest.fixture
def element_data(parser):
    """Loads required FEM data: material, geometry, mesh, and loads."""
    try:
        material_props = parser.material_parser(BASE_DIR / "material.txt")
        geometry_data = parser.geometry_parser(BASE_DIR / "geometry.txt")
        mesh_data = parser.mesh_parser(JOB_DIR / "k_2_node_101.txt", BASE_DIR / "geometry.txt")
        loads_array = parser.load_parser(JOB_DIR / "load.txt")
        return material_props, geometry_data, mesh_data, loads_array
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        pytest.fail("Failed to load element test data.")

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

            logging.info(f"{element_class.__name__} Element {element_id}: Ke shape {Ke.shape}")

            # Store Ke in separate log
            matrix_logger.info(f"\nElement {element_id} Ke:\n{Ke}")

            # Validate shape
            assert Ke.shape == (12, 12), (
                f"{element_class.__name__} Element {element_id}: Ke should be (12,12), "
                f"but got shape {Ke.shape}"
            )

            # Symmetry Check
            assert np.allclose(Ke, Ke.T), f"{element_class.__name__} Element {element_id}: Ke is not symmetric!"

            # Positive Definiteness Check
            try:
                np.linalg.cholesky(Ke)
            except np.linalg.LinAlgError:
                pytest.fail(f"{element_class.__name__} Element {element_id}: Ke is not positive definite!")

            # Optional: Convert to sparse for large models
            Ke_sparse = scipy.sparse.csr_matrix(Ke)
            logging.info(f"{element_class.__name__} Element {element_id}: Ke converted to sparse (nnz={Ke_sparse.nnz})")

        except Exception as e:
            logging.error(f"Error testing {element_class.__name__} Element {element_id}: {e}")
            pytest.fail(f"Failed Ke test for {element_class.__name__} Element {element_id}")

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

            logging.info(f"{element_class.__name__} Element {element_id}: Fe shape {Fe.shape}")

            # Store Fe in separate log
            matrix_logger.info(f"\nElement {element_id} Fe:\n{Fe}")

            # Validate shape
            assert Fe.shape == (12,), (
                f"{element_class.__name__} Element {element_id}: Fe should be (12,), "
                f"but got shape {Fe.shape}"
            )

            # Validate that `Fe` includes applied loads
            applied_forces = loads_array[element_id]
            assert np.allclose(Fe, applied_forces, atol=1e-6), (
                f"{element_class.__name__} Element {element_id}: Fe does not match applied loads! "
                f"Expected {applied_forces}, got {Fe}"
            )

        except Exception as e:
            logging.error(f"Error testing {element_class.__name__} Element {element_id}: {e}")
            pytest.fail(f"Failed Fe test for {element_class.__name__} Element {element_id}")
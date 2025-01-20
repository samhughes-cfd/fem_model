"""
Element-Level Finite Element Testing (Ke & Fe)
----------------------------------------------

Tests:
    1. Assembly of element stiffness matrices (Ke).
    2. Assembly of element force vectors (Fe).
    3. Correctness of matrix/vector dimensions.

Run:
    pytest_testing\TEST_ELEMENTS.py
"""

import sys
import os
import logging
import numpy as np
import traceback
import time
from pprint import pprint

# Logging Configuration
LOG_FILE = "element_ke_fe_test.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="w")]
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from pre_processing.element_library.euler_bernoulli.euler_bernoulli import EulerBernoulliBeamElement
    from pre_processing.element_library.timoshenko.timoshenko import TimoshenkoBeamElement
    from workflow_manager.parsing.parser_base import ParserBase
except ImportError as e:
    logging.error(f"Module import failure: {e}")
    logging.error(traceback.format_exc())
    sys.exit(1)

ELEMENT_CLASSES = {
    "EulerBernoulliBeamElement": EulerBernoulliBeamElement,
    "TimoshenkoBeamElement": TimoshenkoBeamElement,
}

def main():
    """Tests assembly of element-level Ke and Fe for multiple elements."""

    JOBS_DIR = os.path.join(PROJECT_ROOT, "jobs")
    BASE_DIR = os.path.join(JOBS_DIR, "base")
    JOB_DIR = os.path.join(JOBS_DIR, "job_0001")
    
    parser = ParserBase()

    try:
        material_props = parser.material_parser(os.path.join(BASE_DIR, "material.txt"))
        geometry_data = parser.geometry_parser(os.path.join(BASE_DIR, "geometry.txt"))
        mesh_data = parser.mesh_parser(os.path.join(JOB_DIR, "k_2_node_101.txt"), os.path.join(BASE_DIR, "geometry.txt"))
        load_file_path = os.path.join(JOB_DIR, "load.txt")
        loads_array = parser.parse_load(load_file_path)
    except Exception as e:
        logging.error("Error parsing input files:")
        logging.error(traceback.format_exc())
        return

    if not all([material_props, geometry_data, mesh_data, mesh_data.get("node_positions"), loads_array.size > 0]):
        logging.error("Missing or invalid input data.")
        return

    try:
        element_type = mesh_data.get("element_types", ["EulerBernoulliBeamElement"])[0]
        ElementClass = ELEMENT_CLASSES.get(element_type)
        if ElementClass is None:
            raise ValueError(f"Unsupported element type: {element_type}")
    except Exception as e:
        logging.error("Error determining element type:")
        logging.error(traceback.format_exc())
        return

    elements_instances = []
    stiffness_matrices = {}
    force_vectors = {}

    logging.info(f"Testing assembly of Ke and Fe for {len(mesh_data['connectivity'])} elements.")

    start_time = time.time()

    for element_id, (node1, node2) in enumerate(mesh_data["connectivity"]):
        try:
            element = ElementClass(
                element_id=element_id,
                material=material_props,
                section_props=geometry_data,
                mesh_data=mesh_data,
                node_positions=mesh_data["node_positions"],
                loads_array=loads_array
            )

            element.element_stiffness_matrix()
            Ke = element.Ke

            element.element_force_vector()
            Fe = element.Fe

            if Ke is None or Fe is None:
                raise ValueError(f"Element {element_id} did not compute Ke or Fe correctly.")

            stiffness_matrices[element_id] = Ke
            force_vectors[element_id] = Fe
            elements_instances.append(element)

            logging.info(f"Element {element_id}: Ke shape {Ke.shape}, Fe shape {Fe.shape}")

            assert Ke.shape == (6, 6), f"Element {element_id}: Ke should be (6,6), got {Ke.shape}."
            assert Fe.shape == (6,), f"Element {element_id}: Fe should be (6,), got {Fe.shape}."

        except Exception as e:
            logging.error(f"Error processing element {element_id}:")
            logging.error(traceback.format_exc())

    total_time = time.time() - start_time

    if not elements_instances:
        logging.error("No valid elements were instantiated. Terminating.")
        return

    logging.info(f"Processed {len(elements_instances)} elements in {total_time:.3f} seconds.")

    print("\n=== ELEMENT STIFFNESS MATRICES (Ke) ===")
    for element_id, Ke in stiffness_matrices.items():
        print(f"\nElement {element_id} Ke:")
        pprint(Ke)
        logging.info(f"Element {element_id} Ke:\n{Ke}")

    print("\n=== ELEMENT FORCE VECTORS (Fe) ===")
    for element_id, Fe in force_vectors.items():
        print(f"\nElement {element_id} Fe:")
        pprint(Fe)
        logging.info(f"Element {element_id} Fe:\n{Fe}")

    logging.info("Element Ke and Fe assembly test complete.")

if __name__ == "__main__":
    main()
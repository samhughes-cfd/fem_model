"""
Test Script for Finite Element Computations (Ke and Fe)
--------------------------------------------------------

This script validates the finite element computation of stiffness matrices (Ke) and 
elemental force vectors (Fe) for different element types.

Execute from the **project root directory**:

    python pre_processing/element_library/TEST_ELEMENTS.py
"""

import sys
import os
import logging
import numpy as np
from pprint import pprint

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("element_test.log", mode="w")
    ]
)

# Ensure the project root directory is included in `sys.path`
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))  # Root of the project

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure 'pre_processing' is correctly added to `sys.path`
PRE_PROCESSING_DIR = os.path.join(PROJECT_ROOT, "pre_processing")
if PRE_PROCESSING_DIR not in sys.path:
    sys.path.insert(0, PRE_PROCESSING_DIR)

# Import Element Classes and Parsing Utilities
try:
    from pre_processing.element_library.euler_bernoulli.euler_bernoulli import EulerBernoulliBeamElement
    from pre_processing.element_library.timoshenko.timoshenko import TimoshenkoBeamElement
    from pre_processing.parsing.parser_base import ParserBase
    from pre_processing.parsing.load_parser import parse_load
except ImportError as e:
    logging.error(f"Module import failure: {e}")
    sys.exit(1)

# Mapping of Element Types
ELEMENT_CLASSES = {
    "EulerBernoulliBeamElement": EulerBernoulliBeamElement,
    "TimoshenkoBeamElement": TimoshenkoBeamElement,
}

def main():
    """
    Executes the finite element validation routine:

        1️⃣ Parse material properties from input files.
        2️⃣ Parse geometric properties from input files.
        3️⃣ Parse mesh connectivity data.
        4️⃣ Parse load data.
        5️⃣ Instantiate and validate element objects.
        6️⃣ Compute and log stiffness matrices (Ke).
        7️⃣ Compute and log elemental force vectors (Fe).
    Any encountered errors are logged and reported.
    """

    # Define paths to the required input files
    JOBS_DIR = os.path.join(PROJECT_ROOT, "jobs")
    BASE_DIR = os.path.join(JOBS_DIR, "base")
    JOB_DIR = os.path.join(JOBS_DIR, "job_0001")
    
    parser = ParserBase()

    # ✅ Step 1: Parse Material Properties
    try:
        material_props = parser.material_parser(os.path.join(BASE_DIR, "material.txt"))
        logging.info("Material properties successfully parsed.")
    except Exception as e:
        logging.error(f"Error parsing material properties: {e}")
        return

    # ✅ Step 2: Parse Geometry Properties
    try:
        geometry_data = parser.geometry_parser(os.path.join(BASE_DIR, "geometry.txt"))
        logging.info("Geometry data successfully parsed.")
    except Exception as e:
        logging.error(f"Error parsing geometry data: {e}")
        return

    # ✅ Step 3: Parse Mesh Data
    try:
        mesh_data = parser.mesh_parser(os.path.join(JOB_DIR, "mesh.txt"), os.path.join(BASE_DIR, "geometry.txt"))
        logging.info("Mesh data successfully parsed.")
    except Exception as e:
        logging.error(f"Error parsing mesh data: {e}")
        return

    # ✅ Step 4: Parse Load Data
    try:
        load_file_path = os.path.join(JOB_DIR, "load.txt")
        loads_array = parse_load(load_file_path)
        logging.info("Load data successfully parsed.")
    except Exception as e:
        logging.error(f"Error parsing load data: {e}")
        return

    # Extract section properties from geometry_data
    section_props = {
        "A": geometry_data.get("A", 0.0),     # Cross-sectional area
        "Iz": geometry_data.get("Iz", 0.0)    # Second moment of area about the z-axis
    }

    # Extract node positions from mesh_data
    node_positions = mesh_data.get("node_positions", np.array([]))  # Ensure this key exists

    # Validate that required data is present
    if section_props["A"] <= 0 or section_props["Iz"] <= 0:
        logging.error("Invalid section properties: 'A' and 'Iz' must be positive.")
        return

    if node_positions.size == 0:
        logging.error("Node positions are missing in mesh data.")
        return

    # Validate loads_array
    if loads_array.size == 0:
        logging.error("Loads array is empty.")
        return

    # Ensure that loads_array has as many rows as there are nodes
    num_nodes = mesh_data.get("node_ids", [])
    if len(loads_array) < len(num_nodes):
        # Expand the loads_array to accommodate all nodes
        expanded_loads = np.zeros((len(num_nodes), 6), dtype=float)
        expanded_loads[:loads_array.shape[0], :] = loads_array
        loads_array = expanded_loads
        logging.warning("Loads array was expanded to match the number of nodes.")

    # ✅ Step 5: Determine the Element Type
    element_type = mesh_data.get("element_types", ["EulerBernoulliBeamElement"])[0]  # Default selection
    ElementClass = ELEMENT_CLASSES.get(element_type)

    if ElementClass is None:
        logging.error(f"Unsupported element type: {element_type}. Available options: {list(ELEMENT_CLASSES.keys())}")
        return

    logging.info(f"Initializing computations for element type: {element_type}")

    # ✅ Step 6: Instantiate Elements and Compute Ke and Fe
    elements_instances = []
    stiffness_matrices = {}
    force_vectors = {}

    for element_id, (node1, node2) in enumerate(mesh_data["connectivity"]):
        try:
            # Instantiate element with correct arguments
            element = ElementClass(
                element_id=element_id,
                material=material_props,
                section_props=section_props,
                mesh_data=mesh_data,
                node_positions=node_positions,
                loads_array=loads_array
            )

            # Compute Ke
            element.element_stiffness_matrix()
            Ke = element.Ke
            stiffness_matrices[element_id] = Ke

            # Compute Fe
            element.element_force_vector()
            Fe = element.Fe
            force_vectors[element_id] = Fe

            elements_instances.append(element)

            logging.info(f"Element {element_id}: Computed Ke (shape {Ke.shape}) and Fe (shape {Fe.shape}).")

        except Exception as e:
            logging.error(f"Error initializing element {element_id}: {e}")

    if not elements_instances:
        logging.error("No valid elements were instantiated. Terminating validation process.")
        return

    logging.info(f"Total number of elements processed: {len(elements_instances)}")

    # ✅ Step 7: Output Results
    print("\n=== ELEMENT STIFFNESS MATRICES (Ke) ===")
    for element_id, Ke in stiffness_matrices.items():
        print(f"\nElement {element_id} Stiffness Matrix (Ke):")
        pprint(Ke)
        logging.info(f"Element {element_id} Ke:\n{Ke}")

    print("\n=== ELEMENT FORCE VECTORS (Fe) ===")
    for element_id, Fe in force_vectors.items():
        print(f"\nElement {element_id} Force Vector (Fe):")
        pprint(Fe)
        logging.info(f"Element {element_id} Fe:\n{Fe}")

    logging.info("Finite element validation complete.")

if __name__ == "__main__":
    main()
"""
A test script to validate parsing routines in `workflow_manager/parsing/`.
Run this file from the **project root directory**:
    
    python -m workflow_manager.parsing.test_parsers
"""

import sys
import os
import logging
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Ensure that the project root (fem_model) is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # parsing directory
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))  # fem_model root

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import parsers using absolute imports (Avoid relative imports)
from workflow_manager.parsing.boundary_condition_parser import parse_boundary_conditions
from workflow_manager.parsing.material_parser import parse_material
from workflow_manager.parsing.mesh_parser import parse_mesh
from workflow_manager.parsing.geometry_parser import parse_geometry
from workflow_manager.parsing.solver_parser import parse_solver
from workflow_manager.parsing.load_parser import parse_load

# Ensure that all required files exist before parsing
def validate_file(file_path):
    """Check if a file exists before parsing."""
    if not os.path.isfile(file_path):
        logging.error(f"File not found: {file_path}")
        return False
    return True

def main():
    """Runs test cases for all parsing routines."""

    JOBS_DIR = os.path.join(PROJECT_ROOT, "jobs")
    JOB_DIR = os.path.join(JOBS_DIR, "job_0001")  # Example job directory
    BASE_DIR = os.path.join(JOBS_DIR, "base")

    # 1️ Test Boundary Condition Parser
    bc_file = os.path.join(BASE_DIR, "geometry.txt")
    if validate_file(bc_file):
        boundary_data = parse_boundary_conditions(bc_file)
        print("\n--- Boundary Conditions ---")
        pprint(boundary_data)

    # 2️ Test Material Parser
    mat_file = os.path.join(BASE_DIR, "material.txt")
    if validate_file(mat_file):
        materials_data = parse_material(mat_file)
        print("\n--- Materials ---")
        pprint(materials_data)

    # 3️ Test Geometry Parser (Section Geometry)
    section_file = os.path.join(BASE_DIR, "geometry.txt")
    if validate_file(section_file):
        section_data = parse_geometry(section_file)
        print("\n--- Section Geometry ---")
        pprint(section_data)

    # 4️ Test Mesh Parser
    mesh_file = os.path.join(JOB_DIR, "mesh.txt")
    if validate_file(mesh_file) and validate_file(section_file):
        mesh_data = parse_mesh(mesh_file, section_file)
        print("\n--- Mesh Data ---")
        pprint(mesh_data)

    # 5️ Test Solver Parser
    solver_file = os.path.join(BASE_DIR, "solver.txt")
    if validate_file(solver_file):
        try:
            solver_data = parse_solver(solver_file)
            print("\n--- Solver Data ---")
            pprint(solver_data)
        except Exception as e:
            logging.error(f"Error parsing solver file: {e}")

    # 6 Test Load Parser
    load_file = os.path.join(JOB_DIR, "load.txt")
    if validate_file(load_file):
        try:
            load_data = parse_load(load_file)
            print("\n--- Load Data ---")
            pprint(load_data)
        except Exception as e:
            logging.error(f"Error parsing load file: {e}")

if __name__ == "__main__":
    main()
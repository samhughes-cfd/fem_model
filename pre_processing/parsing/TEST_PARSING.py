"""
Test script to validate parsing routines in `pre_processing/parsing/`.

Usage:
    Run this script from the **project root directory**:
    
        python pre_processing/parsing/test_parsers.py

This script will:
    - Validate input files before parsing.
    - Parse:
        ✅ Material Properties
        ✅ Geometry (Section Properties)
        ✅ Mesh Data
        ✅ Load Conditions
    - Print parsed data in a readable format.
    - Log errors and missing files.

🚫 Excluded:
    - Boundary Conditions
    - Solver Settings
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

# Ensure project root (fem_model) is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Current directory (parsing)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))  # fem_model root

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import parsers (excluding boundary conditions and solver parsers)
try:
    from pre_processing.parsing.material_parser import parse_material
    from pre_processing.parsing.geometry_parser import parse_geometry
    from pre_processing.parsing.mesh_parser import parse_mesh
    from pre_processing.parsing.load_parser import parse_load
except ImportError as e:
    logging.error(f"Failed to import parsing modules: {e}")
    sys.exit(1)  # Exit if imports fail

def validate_file(file_path):
    """
    Checks if a given file exists before parsing.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if file exists, False otherwise.
    """
    if not os.path.isfile(file_path):
        logging.error(f"File not found: {file_path}")
        return False
    return True

def main():
    """
    Runs test cases for all parsing routines except boundary conditions and solver settings.

    Tests:
      ✅ Material properties
      ✅ Geometry (section properties)
      ✅ Mesh data
      ✅ Load conditions
    """

    JOBS_DIR = os.path.join(PROJECT_ROOT, "jobs")
    JOB_DIR = os.path.join(JOBS_DIR, "job_0001")  # Example job directory
    BASE_DIR = os.path.join(JOBS_DIR, "base")

    # ✅ 1️⃣ Test Material Parser
    mat_file = os.path.join(BASE_DIR, "material.txt")
    if validate_file(mat_file):
        try:
            materials_data = parse_material(mat_file)
            print("\n--- Materials ---")
            pprint(materials_data)
        except Exception as e:
            logging.error(f"Error parsing material properties: {e}")

    # ✅ 2️⃣ Test Geometry Parser (Section Geometry)
    section_file = os.path.join(BASE_DIR, "geometry.txt")
    if validate_file(section_file):
        try:
            section_data = parse_geometry(section_file)
            print("\n--- Section Geometry ---")
            pprint(section_data)
        except Exception as e:
            logging.error(f"Error parsing section geometry: {e}")

    # ✅ 3️⃣ Test Mesh Parser
    mesh_file = os.path.join(JOB_DIR, "mesh.txt")
    if validate_file(mesh_file) and validate_file(section_file):
        try:
            mesh_data = parse_mesh(mesh_file, section_file)
            print("\n--- Mesh Data ---")
            pprint(mesh_data)
        except Exception as e:
            logging.error(f"Error parsing mesh: {e}")

    # ✅ 4️⃣ Test Load Parser
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
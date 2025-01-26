# pre_processing/parsing/base_parser.py

# ensure, cd C:\Users\samea\Desktop\fem_model
# run, python -m pre_processing.parsing.base_parser

import logging
import os
import sys
from .geometry_parser import parse_geometry
from .load_parser import parse_load
from .material_parser import parse_material
from .mesh_parser import parse_mesh
from .solver_parser import parse_solver

script_dir = os.path.dirname(os.path.abspath(__file__))
fem_model_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if fem_model_root not in sys.path:
    sys.path.insert(0, fem_model_root)

class ParserBase:
    """
    A centralized parser class for bracket-based FEM input files.

    Parses:
      - Geometry
      - Material properties
      - Load conditions
      - Mesh data
      - Solver settings

    Returns:
      - NumPy arrays (geometry, material, load, solver)
      - Tuples of NumPy arrays (mesh)
    """

    def __init__(self):
        """Initialize an empty ParserBase instance."""
        self.logger = logging.getLogger("ParserBase")

    @staticmethod
    def _validate_file(file_path, parser_name):
        """Helper function to check if a file exists before parsing."""
        if not os.path.exists(file_path):
            logging.error(f"[{parser_name}] File not found: {file_path}")
            raise FileNotFoundError(f"{parser_name} file not found: {file_path}")

    def geometry_parser(self, geometry_file_path):
        """Parses the geometry file and returns a NumPy array (1, 20)."""
        self._validate_file(geometry_file_path, "Geometry")
        try:
            return parse_geometry(geometry_file_path)
        except Exception as e:
            self.logger.error(f"[Geometry] Parsing failed: {e}")
            raise

    def material_parser(self, material_file_path):
        """Parses the material file and returns a NumPy array (1, 4)."""
        self._validate_file(material_file_path, "Material")
        try:
            return parse_material(material_file_path)
        except Exception as e:
            self.logger.error(f"[Material] Parsing failed: {e}")
            raise

    def load_parser(self, load_file_path):
        """Parses the load file and returns a NumPy array (N, 9)."""
        self._validate_file(load_file_path, "Load")
        try:
            return parse_load(load_file_path)
        except Exception as e:
            self.logger.error(f"[Load] Parsing failed: {e}")
            raise

    def mesh_parser(self, mesh_file_path):
        """Parses the mesh file and returns a tuple of NumPy arrays."""
        self._validate_file(mesh_file_path, "Mesh")
        try:
            return parse_mesh(mesh_file_path)
        except Exception as e:
            self.logger.error(f"[Mesh] Parsing failed: {e}")
            raise

    def solver_parser(self, solver_file_path):
        """Parses the solver file and returns a NumPy array (3,)."""
        self._validate_file(solver_file_path, "Solver")
        try:
            return parse_solver(solver_file_path)  # Returns np.ndarray of shape (3,)
        except Exception as e:
            self.logger.error(f"[Solver] Parsing failed: {e}")
            raise


# ------------------------------------------------------
# ✅ Standalone execution for direct testing of ParserBase
# ------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = ParserBase()

    # Sample test file paths (adjust if necessary)
    base_path = "jobs/base"
    job_path = "jobs/job_0001"

    test_files = {
        "geometry": os.path.join(base_path, "geometry.txt"),
        "material": os.path.join(base_path, "material.txt"),
        "load": os.path.join(job_path, "load.txt"),
        "mesh": os.path.join(job_path, "mesh.txt"),
        "solver": os.path.join(base_path, "solver.txt"),
    }

    # Run all parsers and print outputs
    try:
        print("\n--- Geometry Data ---\n", parser.geometry_parser(test_files["geometry"]))
        print("\n--- Material Data ---\n", parser.material_parser(test_files["material"]))
        print("\n--- Load Data ---\n", parser.load_parser(test_files["load"]))
        mesh_data = parser.mesh_parser(test_files["mesh"])
        print("\n--- Mesh Data ---\n", mesh_data)
        print("\n--- Solver Data ---\n", parser.solver_parser(test_files["solver"]))
    except Exception as e:
        logging.error(f"Error during standalone parsing test: {e}")
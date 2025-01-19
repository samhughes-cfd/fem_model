# workflow_manager\parse_settings.py

import logging
import os

# Import your bracket-based parser functions
from parsing.geometry_parser import parse_geometry
from parsing.load_parser import parse_load
from parsing.boundary_condition_parser import parse_boundary_conditions
from parsing.material_parser import parse_material
from parsing.mesh_parser import parse_mesh
from parsing.solver_parser import parse_solver

class ParserBase:
    """
    A centralized parser class for bracket-based FEM input files.
    
    Parses:
      - Geometry
      - Material properties
      - Load conditions
      - Boundary conditions
      - Mesh data
      - Solver settings

    Returns:
      - Dicts (geometry, material, mesh, solver)
      - NumPy arrays (loads, boundary conditions)
    """

    def __init__(self):
        """Initialize an empty ParserBase instance."""
        pass

    @staticmethod
    def _validate_file(file_path, parser_name):
        """Helper function to check if a file exists before parsing."""
        if not os.path.exists(file_path):
            logging.error(f"[{parser_name}] File not found: {file_path}")
            raise FileNotFoundError(file_path)

    def geometry_parser(self, geometry_file_path):
        """Parses the geometry file and returns a dictionary."""
        self._validate_file(geometry_file_path, "Geometry")
        return parse_geometry(geometry_file_path)

    def material_parser(self, material_file_path):
        """Parses the material file and returns a dictionary."""
        self._validate_file(material_file_path, "Material")
        return parse_material(material_file_path)

    def load_parser(self, load_file_path):
        """Parses the load file and returns a NumPy array (max_node_id, 6)."""
        self._validate_file(load_file_path, "Load")
        return parse_load(load_file_path)

    def boundary_conditions_parser(self, bc_file_path):
        """Parses the boundary conditions file and returns a NumPy array (max_node_id, 6)."""
        self._validate_file(bc_file_path, "Boundary Conditions")
        return parse_boundary_conditions(bc_file_path)

    def mesh_parser(self, mesh_file_path, geometry_file_path):
        """Parses the mesh file and returns a dictionary with element and node data."""
        self._validate_file(mesh_file_path, "Mesh")
        self._validate_file(geometry_file_path, "Geometry")
        return parse_mesh(mesh_file_path, geometry_file_path)

    def solver_parser(self, solver_file_path):
        """Parses the solver file and returns a dictionary {solver_type: solver_name}."""
        self._validate_file(solver_file_path, "Solver")
        return parse_solver(solver_file_path)
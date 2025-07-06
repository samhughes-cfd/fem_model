import os
import re
import numpy as np
import logging

class BoundaryConditionParser:
    def __init__(self, displacement_file, velocity_file, point_load_file, distributed_load_file):
        self.displacement_file = displacement_file
        self.velocity_file = velocity_file
        self.point_load_file = point_load_file
        self.distributed_load_file = distributed_load_file

    def parse_displacement_conditions(self):
        return self._parse_dirichlet_file(self.displacement_file, "[Prescribed Displacement]", "Displacement")

    def parse_velocity_conditions(self):
        return self._parse_dirichlet_file(self.velocity_file, "[Prescribed Velocity]", "Velocity")

    def parse_point_load(self):
        return self._parse_neumann_file(self.point_load_file, "[Point load]", "Point Load")

    def parse_distributed_load(self):
        return self._parse_neumann_file(self.distributed_load_file, "[Distributed load]", "Distributed Load")

    def parse_all(self):
        return {
            'dirichlet': {
                'displacement': self.parse_displacement_conditions(),
                'velocity': self.parse_velocity_conditions()
            },
            'neumann': {
                'point': self.parse_point_load(),
                'distributed': self.parse_distributed_load()
            }
        }

    def _parse_dirichlet_file(self, file_path, section_header, label):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[{label}] File not found: {file_path}")
        logging.info(f"[{label}] Reading file: {file_path}")

        pattern = re.compile(section_header, re.IGNORECASE)
        current_section = False

        ids, node_ids, dofs, values, types = [], [], [], [], []

        with open(file_path, 'r') as f:
            for line_number, raw_line in enumerate(f, 1):
                line = raw_line.split('#')[0].strip()
                if not line:
                    continue
                if pattern.match(line):
                    current_section = True
                    continue
                if not current_section:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    logging.warning(f"[{label}] Line {line_number}: Incomplete. Skipping.")
                    continue

                try:
                    ids.append(int(parts[0]))
                    node_ids.append(int(parts[1]))
                    dofs.append(parts[2].upper())
                    values.append(float(parts[3]))
                    types.append(parts[4].lower())
                except Exception as e:
                    logging.warning(f"[{label}] Line {line_number}: {e}. Skipping.")

        return {
            'id': np.array(ids, dtype=int),
            'node_id': np.array(node_ids, dtype=int),
            'dof': np.array(dofs, dtype=str),
            'value': np.array(values, dtype=float),
            'type': np.array(types, dtype=str)
        }

    def _parse_neumann_file(self, file_path, section_header, label):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[{label}] File not found: {file_path}")
        logging.info(f"[{label}] Reading file: {file_path}")

        pattern = re.compile(section_header, re.IGNORECASE)
        current_section = False
        data = []

        with open(file_path, 'r') as f:
            for line_number, raw_line in enumerate(f, 1):
                line = raw_line.split('#')[0].strip()
                if not line:
                    continue
                if pattern.match(line):
                    current_section = True
                    continue
                if not current_section:
                    continue

                parts = line.split()
                if len(parts) != 9:
                    logging.warning(f"[{label}] Line {line_number}: Expected 9 values, got {len(parts)}. Skipping.")
                    continue

                try:
                    numeric = [float(p) for p in parts]
                    data.append(numeric)
                except ValueError as e:
                    logging.warning(f"[{label}] Line {line_number}: {e}. Skipping.")

        if not data:
            logging.warning(f"[{label}] No valid load entries found.")
            return {
                'id': np.empty((0,), dtype=int),
                'node_id': np.empty((0,), dtype=int),
                'coordinates': np.empty((0, 3), dtype=float),
                'forces': np.empty((0, 3), dtype=float)
            }

        array = np.array(data)
        return {
            'id': array[:, 0].astype(int),
            'node_id': array[:, 1].astype(int),
            'coordinates': array[:, 2:5],
            'forces': array[:, 5:8]
        }

    def _parse_point_load(file_path: str) -> dict:
        """
        Parses a point load file and returns all values as NumPy arrays, suitable for vectorized use.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Point load] File not found: {file_path}")

        logging.info(f"[Point load] Reading file: {file_path}")
    
        header_pattern = re.compile(r"^\[Point load\]$", re.IGNORECASE)
        current_section = False

        ids = []
        node_ids = []
        coords = []
        forces = []
        moments = []

        with open(file_path, 'r') as f:
            for line_number, raw_line in enumerate(f, 1):
                line = raw_line.split('#')[0].strip()
                if not line:
                    continue

                if header_pattern.match(line):
                    logging.debug(f"[Point load] Section start detected at line {line_number}")
                    current_section = True
                    continue

                if not current_section:
                    continue

                parts = line.split()
                if len(parts) != 9:
                    logging.warning(f"[Point load] Line {line_number}: Expected 9 values, found {len(parts)}. Skipping.")
                    continue

                try:
                    values = list(map(float, parts))
                    ids.append(values[0])
                    node_ids.append(values[1])
                    coords.append(values[2:5])
                    forces.append(values[5:8])
                    moments.append([values[8], 0.0, 0.0])  # Extend to full vector if needed
                except ValueError as e:
                    logging.warning(f"[Point load] Line {line_number}: Could not parse floats. Skipping. Error: {e}")
                    continue

        if not ids:
            logging.error("[Point load] No valid entries found. Returning empty arrays.")
            return {
                'id': np.array([]),
                'node_id': np.array([]),
                'coordinates': np.empty((0, 3)),
                'force': np.empty((0, 3)),
                'moment': np.empty((0, 3))
            }

        return {
            'id': np.array(ids, dtype=int),
            'node_id': np.array(node_ids, dtype=int),
            'coordinates': np.array(coords, dtype=float),
            'force': np.array(forces, dtype=float),
            'moment': np.array(moments, dtype=float)
        }

    def _parse_distributed_load(file_path: str) -> dict:
        """
        Parses a distributed load file and returns all values as NumPy arrays in a structured dictionary.
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Distributed load] File not found: {file_path}")

        logging.info(f"[Distributed load] Reading file: {file_path}")

        header_pattern = re.compile(r"^\[Distributed load\]$", re.IGNORECASE)
        current_section = False

        ids = []
        node_ids = []
        coords = []
        forces = []
        moments = []

        with open(file_path, 'r') as f:
            for line_number, raw_line in enumerate(f, 1):
                line = raw_line.split('#')[0].strip()
                if not line:
                    continue

                if header_pattern.match(line):
                    logging.debug(f"[Distributed load] Section start detected at line {line_number}")
                    current_section = True
                    continue

                if not current_section:
                    continue

                parts = line.split()
                if len(parts) != 9:
                    logging.warning(f"[Distributed load] Line {line_number}: Expected 9 values, got {len(parts)}. Skipping.")
                    continue

                try:
                    values = list(map(float, parts))
                    ids.append(values[0])
                    node_ids.append(values[1])
                    coords.append(values[2:5])
                    forces.append(values[5:8])
                    moments.append([values[8], 0.0, 0.0])  # Extend to (Mx, My, Mz) if needed
                except ValueError as e:
                    logging.warning(f"[Distributed load] Line {line_number}: Parsing error. Skipping. Error: {e}")
                    continue

        if not ids:
            logging.error("[Distributed load] No valid entries found. Returning empty arrays.")
            return {
                'id': np.array([]),
                'node_id': np.array([]),
                'coordinates': np.empty((0, 3)),
                'force': np.empty((0, 3)),
                'moment': np.empty((0, 3))
            }

        return {
            'id': np.array(ids, dtype=int),
            'node_id': np.array(node_ids, dtype=int),
            'coordinates': np.array(coords, dtype=float),
            'force': np.array(forces, dtype=float),
            'moment': np.array(moments, dtype=float)
        }
    
    def parse_displacements(file_path: str) -> dict:
        """
        Parses Dirichlet boundary conditions for displacement from a structured text file,
        returning all values as NumPy arrays in a structured dictionary.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Prescribed Displacement] File not found: {file_path}")

        logging.info(f"[Prescribed Displacement] Reading file: {file_path}")

        header_pattern = re.compile(r"^\[Boundary Conditions - Displacement\]$", re.IGNORECASE)
        current_section = False

        ids = []
        node_ids = []
        dofs = []
        values = []
        types = []

        with open(file_path, 'r') as f:
            for line_number, raw_line in enumerate(f, 1):
                line = raw_line.split('#')[0].strip()
                if not line:
                    continue

                if header_pattern.match(line):
                    logging.debug(f"[Prescribed Displacement] Section start detected at line {line_number}")
                    current_section = True
                    continue

                if not current_section:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    logging.warning(f"[Prescribed Displacement] Line {line_number}: Expected at least 5 entries, got {len(parts)}. Skipping.")
                    continue

                try:
                    ids.append(int(parts[0]))
                    node_ids.append(int(parts[1]))
                    dofs.append(parts[2].upper())
                    values.append(float(parts[3]))
                    types.append(parts[4].lower())
                except (ValueError, IndexError) as e:
                    logging.warning(f"[Prescribed Displacement] Line {line_number}: Parsing error: {e}. Skipping.")
                    continue

        if not ids:
            logging.error("[Prescribed Displacement] No valid entries found. Returning empty arrays.")
            return {
                'id': np.array([], dtype=int),
                'node_id': np.array([], dtype=int),
                'dof': np.array([], dtype=str),
                'value': np.array([], dtype=float),
                'type': np.array([], dtype=str)
            }

        return {
            'id': np.array(ids, dtype=int),
            'node_id': np.array(node_ids, dtype=int),
            'dof': np.array(dofs, dtype=str),
            'value': np.array(values, dtype=float),
            'type': np.array(types, dtype=str)
        }
    
    def parse_velocity_conditions(file_path: str) -> dict:
        """
        Parses Dirichlet boundary conditions for velocity from a structured text file,
        returning all values as NumPy arrays in a structured dictionary.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Prescribed Velocity] File not found: {file_path}")

        logging.info(f"[Prescribed Velocity] Reading file: {file_path}")

        header_pattern = re.compile(r"^\[Prescribed Velocity\]$", re.IGNORECASE)
        current_section = False

        ids = []
        node_ids = []
        dofs = []
        values = []
        types = []

        with open(file_path, 'r') as f:
            for line_number, raw_line in enumerate(f, 1):
                line = raw_line.split('#')[0].strip()
                if not line:
                    continue

                if header_pattern.match(line):
                    logging.debug(f"[Prescribed Velocity] Section start detected at line {line_number}")
                    current_section = True
                    continue

                if not current_section:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    logging.warning(f"[Prescribed Velocity] Line {line_number}: Expected at least 5 entries, got {len(parts)}. Skipping.")
                    continue

                try:
                    ids.append(int(parts[0]))
                    node_ids.append(int(parts[1]))
                    dofs.append(parts[2].upper())
                    values.append(float(parts[3]))
                    types.append(parts[4].lower())
                except (ValueError, IndexError) as e:
                    logging.warning(f"[Prescribed Velocity] Line {line_number}: Parsing error: {e}. Skipping.")
                    continue

        if not ids:
            logging.error("[Prescribed Velocity] No valid entries found. Returning empty arrays.")
            return {
                'id': np.array([], dtype=int),
                'node_id': np.array([], dtype=int),
                'dof': np.array([], dtype=str),
                'value': np.array([], dtype=float),
                'type': np.array([], dtype=str)
            }

        return {
            'id': np.array(ids, dtype=int),
            'node_id': np.array(node_ids, dtype=int),
            'dof': np.array(dofs, dtype=str),
            'value': np.array(values, dtype=float),
            'type': np.array(types, dtype=str)
        }
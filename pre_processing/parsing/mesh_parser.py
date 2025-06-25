# pre_processing/parsing/mesh_parser.py

import logging
import numpy as np
import os
import re
import sys
import traceback
from typing import Dict, Tuple, List, Set
from textwrap import dedent

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)

# Element type registry with validation support
_ELEMENT_REGISTRY = {
    'EulerBernoulliBeamElement3D',
    'TimoshenkoBeamElement3D', 
    'LevinsonBeamElement3D'
}

# Strict header pattern matching
_HEADER_PATTERN = re.compile(
    r'\[node_ids\][\s\t]+\[x\][\s\t]+\[y\][\s\t]+\[z\][\s\t]+\[connectivity\][\s\t]+\[element_type\]',
    re.IGNORECASE
)

def parse_mesh(mesh_file_path: str, 
              element_registry: Set[str] = None,
              strict_element_check: bool = True) -> Dict:
    """
    Parses a structured mesh file with strict format validation and type checking.

    =============================
    Mesh Properties Mapping
    =============================

    Property             Key in Dictionary         Data Type             Shape     Indexing    Units  
    ------------------------------------------------------------------------------------------------
    Node IDs             `node_ids`               `np.ndarray[int]`      (N,)      0-based      -      
    Node Positions       `node_coordinates`       `np.ndarray[float]`    (N, 3)    0-based      [m] 
    Connectivity         `connectivity`           `np.ndarray[int]`      (M, 2)    0-based      -      
    Element IDs          `element_ids`            `np.ndarray[int]`      (M,)      0-based      -      
    Element Lengths      `element_lengths`        `np.ndarray[float]`    (M,)      0-based      [m] 
    Element Types        `element_types`          `np.ndarray[str]`      (M,)      0-based      -   

    Parameters
    ----------
    mesh_file_path : str
        Path to structured mesh file with required header format:
        Line 1: [Mesh]
        Line 2: [node_ids]     [x]            [y]         [z]         [connectivity]      [element_type]
    element_registry : Set[str], optional
        Custom element type registry (default: built-in beam elements)
    strict_element_check : bool, optional
        Whether to raise errors for unknown element types (default: True)

    Returns
    -------
    dict
        Dictionary containing:
        - 'node_ids': np.ndarray[int64] of original node IDs (1-based from file)
        - 'node_coordinates': np.ndarray[float64] of XYZ coordinates (N×3)
        - 'connectivity': np.ndarray[int64] of element connections (M×2)
        - 'element_ids': np.ndarray[int64] of generated element IDs (0-based)
        - 'element_lengths': np.ndarray[float64] of element lengths
        - 'element_types': np.ndarray[object] of element type strings

    Raises
    ------
    FileNotFoundError
        If mesh file doesn't exist
    ValueError
        For header/format issues, duplicate nodes, invalid connectivity, or unknown elements (strict mode)
    TypeError
        If numeric conversions fail
    """
    
    # Validate file existence and headers first
    _validate_file_exists(mesh_file_path)
    _validate_header_structure(mesh_file_path)

    # Set up element registry
    element_registry = element_registry or _ELEMENT_REGISTRY

    # Two-pass parsing
    node_id_map, node_coords = _parse_nodes(mesh_file_path)
    connectivity, element_types = _process_connectivity(
        mesh_file_path,
        node_id_map,
        element_registry,
        strict_element_check
    )

    return _finalize_mesh_data(node_id_map, node_coords, connectivity, element_types)

def _validate_file_exists(path: str) -> None:
    """Ensure target file exists and is readable."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Mesh file not found: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Insufficient permissions to read: {path}")

def _validate_header_structure(path: str) -> None:
    """Validate mandatory header lines with strict formatting."""
    with open(path, 'r') as f:
        # Validate first line
        first_line = f.readline().strip()
        if not re.match(r'^\[mesh\]$', first_line, re.IGNORECASE):
            raise ValueError("First line must be exactly '[Mesh]' (case-insensitive)")
        
        # Validate second line
        second_line = f.readline().strip()
        if not _HEADER_PATTERN.fullmatch(second_line):
            raise ValueError(
                "Invalid header format. Second line must be:\n"
                "[node_ids]     [x]            [y]         [z]         [connectivity]      [element_type]"
            )

def _parse_nodes(path: str) -> Tuple[Dict[int, int], List[List[float]]]:
    """First pass: Parse all node definitions with validation."""
    node_id_map = {}
    node_coords = []
    
    with open(path, 'r') as f:
        # Skip headers
        f.readline(), f.readline()
        
        for line_num, line in enumerate(f, start=3):
            clean_line = re.sub(r'#.*', '', line).strip()
            if not clean_line:
                continue

            # Parse node line
            parts = re.split(r'\s+', clean_line, maxsplit=4)
            if len(parts) < 4:
                logging.warning(f"Skipping invalid node format at line {line_num}")
                continue

            try:
                node_id = int(parts[0])
                if node_id in node_id_map:
                    raise ValueError(f"Duplicate node ID {node_id} at line {line_num}")
                
                # Convert coordinates to float64
                coords = [
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3])
                ]
                if not all(np.isfinite(coords)):
                    raise ValueError(f"Non-finite coordinate at line {line_num}")

                node_id_map[node_id] = len(node_coords)
                node_coords.append(coords)

            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid node data at line {line_num}") from e

    return node_id_map, node_coords

def _process_connectivity(path: str,
                         node_id_map: Dict[int, int],
                         element_registry: Set[str],
                         strict_mode: bool) -> Tuple[np.ndarray, List[str]]:
    """Second pass: Process connectivity and element types."""
    connectivity = []
    element_types = []
    
    with open(path, 'r') as f:
        # Skip headers
        f.readline(), f.readline()
        
        for line_num, line in enumerate(f, start=3):
            clean_line = re.sub(r'#.*', '', line).strip()
            if not clean_line:
                continue

            # Extract connectivity and element type
            conn_match = re.search(r'\((\d+)\s*,\s*(\d+)\)', clean_line)
            if not conn_match:
                continue  # Skip lines without connectivity

            # Extract element type from end of line
            elem_type = 'Unknown'
            type_part = clean_line.split(')')[-1].strip()
            if type_part:
                elem_type = re.split(r'\s+', type_part)[0]

            # Validate element type
            if elem_type not in element_registry:
                msg = f"Unknown element type '{elem_type}' at line {line_num}"
                if strict_mode:
                    raise ValueError(f"{msg}\nValid types: {sorted(element_registry)}")
                logging.warning(msg)

            # Process connectivity
            try:
                start_id = int(conn_match.group(1))
                end_id = int(conn_match.group(2))
                
                start_idx = node_id_map[start_id]
                end_idx = node_id_map[end_id]
                
                connectivity.append((start_idx, end_idx))
                element_types.append(elem_type)

            except KeyError as e:
                missing = int(e.args[0])
                raise ValueError(
                    f"Undefined node {missing} in connectivity at line {line_num}"
                ) from None
            except ValueError as e:
                raise ValueError(f"Invalid connectivity format at line {line_num}") from e

    return np.array(connectivity, dtype=np.int64), np.array(element_types, dtype=object)

def _finalize_mesh_data(node_id_map: Dict[int, int],
                       node_coords: List[List[float]],
                       connectivity: np.ndarray,
                       element_types: np.ndarray) -> Dict:
    """Validate and structure final mesh data with numpy arrays."""
    # Convert to numpy arrays with explicit types
    node_ids = np.array(list(node_id_map.keys()), dtype=np.int64)
    node_coords_array = np.array(node_coords, dtype=np.float64)
    
    # Validate connectivity indices
    max_node_idx = len(node_ids) - 1
    if connectivity.size > 0:
        invalid_mask = (connectivity < 0) | (connectivity > max_node_idx)
        if np.any(invalid_mask):
            invalid_nodes = np.unique(connectivity[invalid_mask])
            raise ValueError(
                f"Invalid node indices in connectivity: {invalid_nodes.tolist()}\n"
                f"Valid node indices: 0-{max_node_idx}"
            )

    return {
        'node_ids': node_ids,
        'node_coordinates': node_coords_array,
        'connectivity': connectivity,
        'element_ids': np.arange(len(connectivity), dtype=np.int64),
        'element_lengths': _compute_element_lengths(connectivity, node_coords_array),
        'element_types': element_types
    }

def _compute_element_lengths(connectivity: np.ndarray,
                            node_coords: np.ndarray) -> np.ndarray:
    """Compute element lengths using vectorized Euclidean distance."""
    if connectivity.size == 0:
        return np.empty(0, dtype=np.float64)
    
    start_points = node_coords[connectivity[:, 0]]
    end_points = node_coords[connectivity[:, 1]]
    return np.linalg.norm(end_points - start_points, axis=1).astype(np.float64)

# if __name__ == "__main__":

#     def print_array_details(name, arr, units=None, indexing=None):
#         """Print formatted array metadata and full contents"""
#         print(f"\n{' ' + name + ' ':-^80}")
#         print(f"Data Type: {type(arr).__name__} ({arr.dtype})")
#         print(f"Shape: {arr.shape}")
#         print(f"Units: {units or '-'}")
#         print(f"Indexing: {indexing or '0-based'}")
#         print("\nFull Data:")
#         with np.printoptions(
#             threshold=10,  # Show up to 10 elements before truncating
#             edgeitems=2,   # Show first/last 2 elements when truncated
#             linewidth=80,
#             suppress=True,  # Suppress scientific notation
#             formatter={'float': lambda x: f"{x:.6f}"}  # 6 decimal places for floats
#         ):
#             print(arr)

#     def validate_mesh_integrity(mesh_dict):
#         """Enhanced validation of mesh data structure"""
#         # Check array lengths
#         if len(mesh_dict['element_ids']) != len(mesh_dict['element_types']):
#             raise ValueError("Element IDs and types count mismatch")
            
#         if len(mesh_dict['connectivity']) != len(mesh_dict['element_ids']):
#             raise ValueError("Connectivity and element ID count mismatch")

#         # Validate node indices in connectivity
#         max_node_idx = len(mesh_dict['node_ids']) - 1
#         connectivity = mesh_dict['connectivity']
#         if connectivity.size > 0:
#             invalid = connectivity[(connectivity < 0) | (connectivity > max_node_idx)]
#             if invalid.size > 0:
#                 invalid_nodes = np.unique(invalid)
#                 raise ValueError(
#                     f"Invalid node indices in connectivity: {invalid_nodes.tolist()}\n"
#                     f"Valid node indices: 0-{max_node_idx}"
#                 )

#     # Execution flow
#     if len(sys.argv) > 1:
#         input_file = sys.argv[1]
#     else:
#         input_file = os.path.join("jobs", "job_0009", "mesh.txt")  # Default test file

#     if not os.path.exists(input_file):
#         logging.error(f"Mesh file '{input_file}' not found.")
#         sys.exit(1)

#     try:
#         print(f"\n{' INITIALIZING MESH PARSER ':=^80}")
#         mesh_dict = parse_mesh(input_file)

#         # Perform comprehensive validation
#         validate_mesh_integrity(mesh_dict)

#         print("\n" + "="*80)
#         print("                          Parsed Mesh Data Structure")
#         print("="*80)

#         # General information
#         num_nodes = len(mesh_dict['node_ids'])
#         num_elements = len(mesh_dict['element_ids'])
#         print(f"\n● Total Nodes: {num_nodes}")
#         print(f"● Total Elements: {num_elements}")
        
#         if num_elements != num_nodes - 1:
#             print(f"⚠ Note: Element count ({num_elements}) differs from node count - 1 ({num_nodes-1})")

#         # Detailed array information
#         print_array_details(
#             name="Node IDs",
#             arr=mesh_dict['node_ids'],
#             units="-",
#             indexing="Original file IDs (1-based)"
#         )

#         print_array_details(
#             name="Node Coordinates",
#             arr=mesh_dict['node_coordinates'],
#             units="meters",
#             indexing="0-based [node_index, xyz]"
#         )

#         print_array_details(
#             name="Element Connectivity",
#             arr=mesh_dict['connectivity'],
#             units="-",
#             indexing="0-based node indices"
#         )

#         print_array_details(
#             name="Element IDs",
#             arr=mesh_dict['element_ids'],
#             units="-",
#             indexing="0-based array indices"
#         )

#         print_array_details(
#             name="Element Lengths",
#             arr=mesh_dict['element_lengths'],
#             units="meters",
#             indexing="0-based array indices"
#         )

#         print_array_details(
#             name="Element Types",
#             arr=mesh_dict['element_types'],
#             units="-",
#             indexing="0-based array indices"
#         )

#         print("\n✅ Mesh parsing completed successfully!")
#         print("="*80 + "\n")

#     except Exception as e:
#         print(f"\n{' PARSING FAILED ':=^80}")
#         logging.error("Critical error: %s", str(e))
#         logging.debug("Full error trace:\n%s", traceback.format_exc())
#         sys.exit(1)
import os
import sys
import logging
import numpy as np
from logging.handlers import RotatingFileHandler
from typing import Tuple, Dict, Any

# --- Adjust Python Path to Include Project Root ---
script_dir = os.path.dirname(os.path.abspath(__file__))
fem_model_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if fem_model_root not in sys.path:
    sys.path.insert(0, fem_model_root)

# --- Import Required Modules ---
from pre_processing.parsing.geometry_parser import parse_geometry
from pre_processing.parsing.mesh_parser import parse_mesh
from pre_processing.parsing.material_parser import parse_material
from pre_processing.parsing.point_load_parser import parse_point_load
from pre_processing.parsing.distributed_load_parser import parse_distributed_load
from pre_processing.element_library.utilities.interpolate_loads import interpolate_loads

# --- Global Logger Placeholder ---
# This will be set in main to the intermediate logger.
logger = None

# --- Logger Configuration for Two Separate Logs ---
def configure_loggers() -> Tuple[logging.Logger, logging.Logger]:
    """
    Configure two loggers:
      - int_logger: Detailed log of intermediate steps and tensor evaluations.
      - final_logger: Final summary log with the computed Fe and Ke for each element.
    """
    log_dir = os.path.join(fem_model_root, "post_processing", "tensor_visualisers")
    os.makedirs(log_dir, exist_ok=True)

    # ----- Intermediate Logger -----
    int_logger = logging.getLogger("IntermediateLogger")
    int_logger.setLevel(logging.DEBUG)
    if int_logger.hasHandlers():
        int_logger.handlers.clear()
    int_file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, "eb_tensor_validation_intermediate.log"),
        mode='a',
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    int_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
    )
    int_file_handler.setFormatter(int_formatter)
    int_logger.addHandler(int_file_handler)

    # Optionally add a console handler for intermediate logs
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    int_logger.addHandler(console_handler)

    # ----- Final Logger -----
    final_logger = logging.getLogger("FinalLogger")
    final_logger.setLevel(logging.INFO)
    if final_logger.hasHandlers():
        final_logger.handlers.clear()
    final_file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, "eb_tensor_validation_final.log"),
        mode='a',
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    final_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    final_file_handler.setFormatter(final_formatter)
    final_logger.addHandler(final_file_handler)

    return int_logger, final_logger

# --- File and Data Validation Functions (unchanged) ---
def validate_file_path(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

def validate_geometry_array(geometry_array: np.ndarray) -> None:
    expected_shape = (1, 20)
    if geometry_array.shape != expected_shape:
        raise ValueError(f"Unexpected geometry array shape: {geometry_array.shape}")

def validate_material_array(material_array: np.ndarray) -> None:
    expected_shape = (1, 4)
    if material_array.shape != expected_shape:
        raise ValueError(f"Unexpected material array shape: {material_array.shape}")

def load_data(base_dir: str, job_dir: str, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray, np.ndarray]:
    logger.info("Loading input data from:\n- Base: %s\n- Job: %s", base_dir, job_dir)
    
    # Required files (geometry, material, solver, mesh)
    required_files = {
        'geometry': os.path.join(base_dir, "geometry.txt"),
        'material': os.path.join(base_dir, "material.txt"),
        'solver': os.path.join(base_dir, "solver.txt"),
        'mesh': os.path.join(job_dir, "mesh.txt")
    }
    
    # Validate required files
    for name, path in required_files.items():
        validate_file_path(path)
        logger.debug("%s file validated: %s", name.capitalize(), path)
    
    # Load required data
    logger.info("Parsing geometry data...")
    geometry_array = parse_geometry(required_files['geometry'])
    logger.info("Parsing material properties...")
    material_array = parse_material(required_files['material'])
    logger.info("Parsing mesh configuration...")
    mesh_dictionary = parse_mesh(required_files['mesh'])
    
    # Load optional distributed/point load files if they exist
    distributed_load_path = os.path.join(job_dir, "distributed_load.txt")
    point_load_path = os.path.join(job_dir, "point_load.txt")
    
    # Initialize load arrays as empty
    distributed_load_array = np.empty((0, 9))
    point_load_array = np.empty((0, 9))
    
    # Parse distributed loads if file exists
    if os.path.exists(distributed_load_path):
        logger.info("Parsing distributed load data...")
        distributed_load_array = parse_distributed_load(distributed_load_path)
    else:
        logger.warning("No distributed_load.txt found in job directory. Proceeding with no distributed loads.")
    
    # Parse point loads if file exists
    if os.path.exists(point_load_path):
        logger.info("Parsing point load data...")
        point_load_array = parse_point_load(point_load_path)
    else:
        logger.warning("No point_load.txt found in job directory. Proceeding with no point loads.")
    
    # Validate geometry and material arrays
    validate_geometry_array(geometry_array)
    validate_material_array(material_array)
    
    # Validate load arrays (if non-empty)
    for load_type, load_arr in [("Distributed", distributed_load_array), ("Point", point_load_array)]:
        if load_arr.size > 0 and load_arr.shape[1] != 9:
            logger.error("Invalid %s load array dimensions: %s (Expected Nx9)", load_type, load_arr.shape)
            raise ValueError(f"{load_type} load array must have 9 columns")
    
    logger.info("All data loaded and validated successfully")
    return geometry_array, material_array, mesh_dictionary, distributed_load_array, point_load_array

def shape_functions(xi: np.ndarray, L: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute shape functions and their derivatives for a 2-node 3D Euler-Bernoulli beam element.
    Returns:
      N_matrix      : (g, 12, 6)
      dN_dxi_matrix : (g, 12, 6)
      d2N_dxi2_matrix : (g, 12, 6)
    """
    xi = np.atleast_1d(xi)
    g = xi.shape[0]
    logger.debug("Step 2: Evaluating shape functions at xi: %s", xi)
    # Axial functions for u_x and theta_x (torsion)
    N1 = 0.5 * (1 - xi)
    N7 = 0.5 * (1 + xi)
    dN1_dxi = -0.5 * np.ones(g)
    dN7_dxi = 0.5 * np.ones(g)
    d2N1_dxi2 = np.zeros(g)
    d2N7_dxi2 = np.zeros(g)
    # Bending in XY plane (u_y, theta_z)
    N2 = 0.25 * (1 - xi)**2 * (2 + xi)
    N8 = 0.25 * (1 + xi)**2 * (2 - xi)
    N3 = (L / 8) * (1 - xi)**2 * (1 + xi)
    N9 = (L / 8) * (1 + xi)**2 * (1 - xi)
    dN2_dxi = 0.5 * (1 - xi) * (2 + xi) - 0.5 * (1 - xi)**2
    dN8_dxi = -0.5 * (1 + xi) * (2 - xi) + 0.5 * (1 + xi)**2
    dN3_dxi = (L / 8) * ((1 - xi)**2 - 2*(1 - xi)*(1 + xi))
    dN9_dxi = (L / 8) * ((1 + xi)**2 - 2*(1 + xi)*(1 - xi))
    d2N2_dxi2 = 1.5 * xi - 0.5
    d2N8_dxi2 = -1.5 * xi + 0.5
    d2N3_dxi2 = (L / 8) * (3 * xi - 1)
    d2N9_dxi2 = (L / 8) * (-3 * xi + 1)
    # Bending in XZ plane (u_z, theta_y) uses same functions as XY
    N4, N10 = N2, N8
    N5, N11 = N3, N9
    dN4_dxi, dN10_dxi = dN2_dxi, dN8_dxi
    dN5_dxi, dN11_dxi = dN3_dxi, dN9_dxi
    d2N4_dxi2, d2N10_dxi2 = d2N2_dxi2, d2N8_dxi2
    d2N5_dxi2, d2N11_dxi2 = d2N3_dxi2, d2N9_dxi2
    # Torsion (theta_x)
    N6, N12 = N1, N7
    dN6_dxi, dN12_dxi = dN1_dxi, dN7_dxi
    d2N6_dxi2, d2N12_dxi2 = d2N1_dxi2, d2N7_dxi2
    # Assemble into (g, 12, 6) matrices.
    row_indices = np.arange(12)
    col_indices = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
    def assemble_matrix(scalars):
        M = np.zeros((g, 12, 6))
        for scalar, row, col in zip(scalars, row_indices, col_indices):
            M[:, row, col] = scalar
        return M
    scalars_N = [N1, N2, N4, N6, N5, N3, N7, N8, N10, N12, N11, N9]
    scalars_dN = [dN1_dxi, dN2_dxi, dN4_dxi, dN6_dxi, dN5_dxi, dN3_dxi,
                  dN7_dxi, dN8_dxi, dN10_dxi, dN12_dxi, dN11_dxi, dN9_dxi]
    scalars_d2N = [d2N1_dxi2, d2N2_dxi2, d2N4_dxi2, d2N6_dxi2, d2N5_dxi2, d2N3_dxi2,
                   d2N7_dxi2, d2N8_dxi2, d2N10_dxi2, d2N12_dxi2, d2N11_dxi2, d2N9_dxi2]
    N_matrix = assemble_matrix(scalars_N)
    dN_dxi_matrix = assemble_matrix(scalars_dN)
    d2N_dxi2_matrix = assemble_matrix(scalars_d2N)
    logger.debug("Step 3: Assembled N_matrix shape: %s", N_matrix.shape)
    logger.debug("Step 3: Assembled dN_dxi_matrix shape: %s", dN_dxi_matrix.shape)
    logger.debug("Step 3: Assembled d2N_dxi2_matrix shape: %s", d2N_dxi2_matrix.shape)
    return N_matrix, dN_dxi_matrix, d2N_dxi2_matrix

def compute_stiffness_matrix(element_idx: int, element_length: float, 
                             geometry_array: np.ndarray, material_array: np.ndarray, 
                             quadrature_order: int = 3) -> np.ndarray:
    """
    Compute the 12x12 stiffness matrix for the element.
    Steps:
      - Step 4: Compute material matrix D.
      - Step 5: Get Gauss points, weights, and Jacobian.
      - Step 6: Evaluate shape function derivatives.
      - Step 7: Compute intermediate tensors and sum over Gauss points.
    """
    logger.info("Step 4: Computing stiffness for element %d with length %e", element_idx, element_length)
    A = geometry_array[0, 1]
    I_x = geometry_array[0, 2]
    I_y = geometry_array[0, 3]
    I_z = geometry_array[0, 4]
    E = material_array[0, 0]
    G = material_array[0, 1]
    EA = E * A
    EI_y = E * I_y
    EI_z = E * I_z
    GJ = G * I_x
    D = np.diag([EA, EI_z, EI_y, GJ, EI_z, EI_y])
    logger.debug("Step 4: Material stiffness matrix D:\n%s", D)
    num_integration_points = max(2, quadrature_order)
    xi_points, weights = np.polynomial.legendre.leggauss(num_integration_points)
    logger.debug("Step 5: Gauss points (xi): %s", xi_points)
    logger.debug("Step 5: Gauss weights: %s", weights)
    detJ = element_length / 2
    logger.debug("Step 5: Jacobian determinant (detJ): %e", detJ)
    _, dN_dxi_matrix, _ = shape_functions(xi_points, element_length)
    logger.debug("Step 6: dN_dxi_matrix shape: %s", dN_dxi_matrix.shape)
    dN_dxi_T_matrix = dN_dxi_matrix.transpose(0, 2, 1)
    logger.debug("Step 6: Transposed dN_dxi_T_matrix shape: %s", dN_dxi_T_matrix.shape)
    # Broken-down integration for stiffness matrix:
    intermediate1 = np.einsum("gik,kl->gil", dN_dxi_matrix, D)
    logger.debug("Step 7: Intermediate tensor 1 (dN_dxi_matrix x D):\n%s", intermediate1)
    intermediate2 = np.einsum("gil,glj->gij", intermediate1, dN_dxi_T_matrix)
    logger.debug("Step 7: Intermediate tensor 2 (Intermediate1 x dN_dxi_T_matrix):\n%s", intermediate2)
    Ke = np.einsum("gij,g->ij", intermediate2, weights) * detJ
    logger.debug("Step 7: Final integrated stiffness matrix (Ke):\n%s", Ke)
    return Ke

def compute_element_force_vector(
    element_idx: int,
    x_start: float,
    x_end: float,
    element_length: float,  # Precomputed from mesh_dictionary
    x_global_end: float,
    distributed_load_array: np.ndarray,
    point_load_array: np.ndarray,
    quadrature_order: int = 3
) -> np.ndarray:
    """
    Compute Fe using precomputed element_length for accuracy and efficiency.
    """
    logger.info("Step 8: Computing force vector for element %d (L=%.3f)", 
                element_idx, element_length)
    Fe = np.zeros(12)
    
    # --- Process Distributed Loads ---
    if distributed_load_array.size > 0:
        num_gauss = max(2, quadrature_order)
        xi_gauss, weights = np.polynomial.legendre.leggauss(num_gauss)
        detJ = element_length / 2
        x_gauss = (xi_gauss + 1) * detJ + x_start  # Global coordinates
        
        # Interpolate distributed loads at Gauss points
        interpolated_forces = interpolate_loads(x_gauss, distributed_load_array)
        
        # Integrate using shape functions
        N_matrix, _, _ = shape_functions(xi_gauss, element_length)
        Fe_distributed = np.einsum("gij,gj,g->i", N_matrix, interpolated_forces, weights) * detJ
        Fe += Fe_distributed
    
    # --- Process Point Loads ---
    if point_load_array.size > 0:
        for load_row in point_load_array:
            x_p = load_row[0]
            F = load_row[3:9]
            
            # Check inclusion based on element position in mesh
            if (x_start <= x_p <= x_end) if np.isclose(x_end, x_global_end) else (x_start <= x_p < x_end):
                xi_p = 2 * (x_p - x_start) / element_length - 1
                N_p, _, _ = shape_functions(np.array([xi_p]), element_length)
                Fe += np.einsum("ij,j->i", N_p[0], F)
    return Fe

# --- Main Function: Using Two Loggers ---
def main() -> None:
    """
    Main function: compute element matrices and output unified log results.
    Intermediate steps and tensor evaluations are logged with the intermediate logger,
    and final element results (Fe and Ke) are logged with the final logger.
    """
    try:
        int_logger, final_logger = configure_loggers()
        global logger
        logger = int_logger  # Set the global logger to the intermediate logger

        logger.info("Step 0: Starting Tensor Validation Test Script")
        
        base_dir = os.path.join(fem_model_root, "jobs", "base")
        job_dir = os.path.join(fem_model_root, "jobs", "job_0003")
        
        geometry_array, material_array, mesh_dictionary, distributed_load_array, point_load_array = load_data(base_dir, job_dir, logger)
        
        # Precompute element metadata
        node_coordinates = mesh_dictionary["node_coordinates"]
        connectivity = mesh_dictionary["connectivity"]
        element_x_start = node_coordinates[connectivity[:, 0], 0]  # x of first node
        element_x_end = node_coordinates[connectivity[:, 1], 0]    # x of second node
        x_global_end = np.max(node_coordinates[:, 0])  # x of max node index
        
        logger.info("Step 1: Computing Element Matrices...")
        results = {}
        for elem_idx in range(len(mesh_dictionary["element_lengths"])):
            x_start = element_x_start[elem_idx]
            x_end = element_x_end[elem_idx]
            elem_length = mesh_dictionary["element_lengths"][elem_idx]
            
            Fe = compute_element_force_vector(
                element_idx=elem_idx,
                x_start=x_start,
                x_end=x_end,
                element_length=elem_length,
                x_global_end=x_global_end,
                distributed_load_array=distributed_load_array,
                point_load_array=point_load_array,
                quadrature_order=3
            )
            
            Ke = compute_stiffness_matrix(elem_idx, elem_length, geometry_array, material_array)
            
            # Store results with element start, length, and end
            results[elem_idx] = {
                "Element Force Vector": Fe.reshape(12, 1),
                "Element Stiffness Matrix": Ke,
                "Element Length": elem_length,
                "Element Start": x_start,
                "Element End": x_end,
                "Connectivity": ", ".join(str(n) for n in connectivity[elem_idx])
            }
            logger.debug("Element %d computed: Fe shape %s, Ke shape %s", elem_idx, Fe.shape, Ke.shape)
        
        # Build final results string to be logged by the final logger
        final_results_str = "\n"
        for elem_idx, result in results.items():
            header = (
                "=" * 191 + "\n" +
                f"Element Index     : {elem_idx}\n" +
                f"Connectivity      : nodes {result['Connectivity']}\n" +
                f"Element Start     : {result['Element Start']:.6e}\n" +
                f"Element Length    : {result['Element Length']:.6e}\n" +
                f"Element End       : {result['Element End']:.6e}\n" +
                "=" * 191 + "\n"
            )
            # Build table header with row and column labels.
            table_header = f"{'Fe0':>18s}"
            for j in range(12):
                table_header += f"{('Ke' + str(j)):>14s}"
            table_header += "\n" + "-" * (4 + 2 + 14 + 3 + 12 * 14) + "\n"
            table_rows = ""
            for i in range(12):
                fe_val = result["Element Force Vector"][i, 0]
                row_line = f"{i:>4d}  {fe_val:+14.6e}   "
                for j in range(12):
                    ke_val = result["Element Stiffness Matrix"][i, j]
                    row_line += f"{ke_val:+14.6e}"
                row_line += "\n"
                table_rows += row_line
            final_results_str += header + table_header + table_rows + "\n\n"
        
        final_logger.info("Final Detailed Results:\n%s", final_results_str)
        logger.info("Step 12: Tensor Validation Test Completed!")
        
    except Exception as e:
        # Log errors with the intermediate logger for full trace details.
        logger.error("Error in main function: %s", e, exc_info=True)
        raise

if __name__ == "__main__":
    main()
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
from pre_processing.parsing.load_parser import parse_load
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

def load_data(base_dir: str, job_dir: str, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
    logger.info("Loading input data from:\n- Base: %s\n- Job: %s", base_dir, job_dir)
    
    file_paths = {
        'geometry': os.path.join(base_dir, "geometry.txt"),
        'material': os.path.join(base_dir, "material.txt"),
        'solver': os.path.join(base_dir, "solver.txt"),
        'mesh': os.path.join(job_dir, "mesh.txt"),
        'load': os.path.join(job_dir, "load.txt")
    }
    
    for name, path in file_paths.items():
        validate_file_path(path)
        logger.debug("%s file validated: %s", name.capitalize(), path)
    
    logger.info("Parsing geometry data...")
    geometry_array = parse_geometry(file_paths['geometry'])
    logger.info("Parsing material properties...")
    material_array = parse_material(file_paths['material'])
    logger.info("Parsing mesh configuration...")
    mesh_dictionary = parse_mesh(file_paths['mesh'])
    logger.info("Parsing load data...")
    load_array = parse_load(file_paths['load'])
    
    validate_geometry_array(geometry_array)
    validate_material_array(material_array)
    
    if load_array.shape[1] != 9:
        logger.error("Invalid load array dimensions: %s (Expected Nx9)", load_array.shape)
        raise ValueError("Load array must have 9 columns")
    
    logger.info("All data loaded and validated successfully")
    return geometry_array, material_array, mesh_dictionary, load_array

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

def compute_element_force_vector(element_idx: int, element_length: float, 
                                 loads_array: np.ndarray, quadrature_order: int = 3) -> np.ndarray:
    """
    Compute the equivalent nodal force vector (12,) for the element.
    Steps:
      - Step 8: Get Gauss points, weights, and physical coordinates.
      - Step 9: Interpolate loads at Gauss points.
      - Step 10: Evaluate the shape function matrix.
      - Step 11: Compute intermediate tensor and sum over Gauss points.
    """
    logger.info("Step 8: Computing force vector for element %d with length %e", element_idx, element_length)
    num_integration_points = max(2, quadrature_order)
    xi_points, weights = np.polynomial.legendre.leggauss(num_integration_points)
    logger.debug("Step 8: Gauss points (xi): %s", xi_points)
    logger.debug("Step 8: Gauss weights: %s", weights)
    detJ = element_length / 2
    logger.debug("Step 8: Jacobian determinant (detJ): %e", detJ)
    x_gauss = (xi_points + 1) * detJ
    logger.debug("Step 8: Physical coordinates at Gauss points (x_gauss): %s", x_gauss)
    interpolated_forces = interpolate_loads(x_gauss, loads_array)
    logger.debug("Step 9: Interpolated forces at Gauss points:\n%s", interpolated_forces)
    N_matrix, _, _ = shape_functions(xi_points, element_length)
    logger.debug("Step 10: Shape function matrix (N_matrix) shape: %s", N_matrix.shape)
    intermediate1 = np.einsum("gij,gj->gi", N_matrix, interpolated_forces)
    logger.debug("Step 11: Intermediate tensor (N_matrix dot interpolated forces):\n%s", intermediate1)
    Fe = np.einsum("gi,g->i", intermediate1, weights) * detJ
    logger.debug("Step 11: Final integrated force vector (Fe):\n%s", Fe)
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
        job_dir = os.path.join(fem_model_root, "jobs", "job_0001")
        geometry_array, material_array, mesh_dictionary, load_array = load_data(base_dir, job_dir, logger)
        
        logger.info("Step 1: Computing Element Matrices...")
        results = {}
        for elem_idx, elem_length in enumerate(mesh_dictionary["element_lengths"]):
            logger.debug("Processing element %d with length: %.4e", elem_idx, elem_length)
            Fe = compute_element_force_vector(elem_idx, elem_length, load_array)
            Ke = compute_stiffness_matrix(elem_idx, elem_length, geometry_array, material_array)
            results[elem_idx] = {
                "Element Force Vector": Fe.reshape(12, 1),
                "Element Stiffness Matrix": Ke,
                "Element Length": elem_length
            }
            logger.debug("Element %d computed: Fe shape %s, Ke shape %s", 
                         elem_idx, Fe.shape, Ke.shape)
        
        # Build final results string to be logged by the final logger
        final_results_str = "\n"
        for elem_idx, result in results.items():
            if "connectivity" in mesh_dictionary:
                conn = mesh_dictionary["connectivity"][elem_idx]
                conn_str = ", ".join(str(n) for n in conn)
            else:
                conn_str = "N/A"
            header = ("=" * 191 + "\n" +
                      f"Element Index     : {elem_idx}\n" +
                      f"Connectivity      : nodes {conn_str}\n" +
                      f"Element Length    : {result['Element Length']:.6e}\n" +
                      "=" * 191 + "\n")
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
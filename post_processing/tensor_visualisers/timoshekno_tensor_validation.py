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
        filename=os.path.join(log_dir, "t_tensor_validation_intermediate.log"),
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

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setLevel(logging.INFO)
    int_logger.addHandler(console_handler)

    # ----- Final Logger -----
    final_logger = logging.getLogger("FinalLogger")
    final_logger.setLevel(logging.INFO)
    if final_logger.hasHandlers():
        final_logger.handlers.clear()
    final_file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, "t_tensor_validation_final.log"),
        mode='a',
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    final_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    final_file_handler.setFormatter(final_formatter)
    final_logger.addHandler(final_file_handler)

    return int_logger, final_logger

# --- File and Data Validation Functions ---
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
    
    geometry_array = parse_geometry(file_paths['geometry'])
    material_array = parse_material(file_paths['material'])
    mesh_dictionary = parse_mesh(file_paths['mesh'])
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
    Compute shape functions and their derivatives for a 2-node 3D Timoshenko beam element.
    Returns:
      N_matrix      : (g, 12, 6)
      dN_dxi_matrix : (g, 12, 6)
      d2N_dxi2_matrix : (g, 12, 6)
    """
    xi = np.atleast_1d(xi)
    g = xi.shape[0]
    logger.debug("Step 2: Evaluating shape functions at xi: %s", xi)
    
    # Linear shape functions for all DOFs
    N1 = 0.5 * (1 - xi)
    N7 = 0.5 * (1 + xi)
    dN1_dxi = -0.5 * np.ones(g)
    dN7_dxi = 0.5 * np.ones(g)
    d2N1_dxi2 = np.zeros(g)
    d2N7_dxi2 = np.zeros(g)
    
    # Assign linear functions to all DOFs
    N2, N8 = N1, N7  # uy
    N3, N9 = N1, N7  # theta_z
    N4, N10 = N1, N7  # uz
    N5, N11 = N1, N7  # theta_y
    N6, N12 = N1, N7  # theta_x
    
    dN2_dxi, dN8_dxi = dN1_dxi, dN7_dxi
    dN3_dxi, dN9_dxi = dN1_dxi, dN7_dxi
    dN4_dxi, dN10_dxi = dN1_dxi, dN7_dxi
    dN5_dxi, dN11_dxi = dN1_dxi, dN7_dxi
    dN6_dxi, dN12_dxi = dN1_dxi, dN7_dxi
    
    d2N2_dxi2, d2N8_dxi2 = d2N1_dxi2, d2N7_dxi2
    d2N3_dxi2, d2N9_dxi2 = d2N1_dxi2, d2N7_dxi2
    d2N4_dxi2, d2N10_dxi2 = d2N1_dxi2, d2N7_dxi2
    d2N5_dxi2, d2N11_dxi2 = d2N1_dxi2, d2N7_dxi2
    d2N6_dxi2, d2N12_dxi2 = d2N1_dxi2, d2N7_dxi2
    
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
    return N_matrix, dN_dxi_matrix, d2N_dxi2_matrix

def compute_stiffness_matrix(element_idx: int, element_length: float, 
                             geometry_array: np.ndarray, material_array: np.ndarray, 
                             quadrature_order: int = 1) -> np.ndarray:  # Reduced integration to mitigate shear locking
    """
    Compute the 12x12 stiffness matrix for the Timoshenko beam element.
    """
    logger.info("Step 4: Computing stiffness for element %d with length %e", element_idx, element_length)
    
    A = geometry_array[0, 1]
    A_y = geometry_array[0, 5]  # Shear area Y
    A_z = geometry_array[0, 6]  # Shear area Z
    I_x = geometry_array[0, 2]
    I_y = geometry_array[0, 3]
    I_z = geometry_array[0, 4]
    
    E = material_array[0, 0]
    G = material_array[0, 1]
    
    EA = E * A
    GA_y = G * A_y
    GA_z = G * A_z
    GJ = G * I_x
    EI_y = E * I_y
    EI_z = E * I_z
    
    D = np.diag([EA, GA_y, GA_z, GJ, EI_y, EI_z])
    logger.debug("Material stiffness matrix D:\n%s", D)
    
    num_integration_points = max(1, quadrature_order)  # Reduced integration
    xi_points, weights = np.polynomial.legendre.leggauss(num_integration_points)
    detJ = element_length / 2
    dxi_dx = 2 / element_length  # Derivative conversion
    
    N_matrix, dN_dxi_matrix, _ = shape_functions(xi_points, element_length)
    dN_dx_matrix = dN_dxi_matrix * dxi_dx  # Shape (g, 12, 6)
    
    col_indices = np.array([0,1,2,3,4,5,0,1,2,3,4,5])
    Ke = np.zeros((12, 12))
    
    for g in range(len(xi_points)):
        N_g = N_matrix[g]
        dN_dx_g = dN_dx_matrix[g]
        weight = weights[g]
        
        B = np.zeros((6, 12))
        
        for j in range(12):
            c = col_indices[j]
            
            if c == 0:  # ux
                B[0, j] = dN_dx_g[j, c]
            elif c == 1:  # uy
                B[1, j] = dN_dx_g[j, c]
            elif c == 2:  # uz
                B[2, j] = dN_dx_g[j, c]
            elif c == 3:  # theta_x
                B[3, j] = dN_dx_g[j, c]
            elif c == 4:  # theta_y
                B[2, j] += N_g[j, c]  # gamma_xz += theta_y
                B[4, j] = dN_dx_g[j, c]
            elif c == 5:  # theta_z
                B[1, j] += -N_g[j, c]  # gamma_xy -= theta_z
                B[5, j] = dN_dx_g[j, c]
        
        Ke += (B.T @ D @ B) * weight * detJ
    
    logger.debug("Element stiffness matrix Ke:\n%s", Ke)
    return Ke

def compute_element_force_vector(element_idx: int, element_length: float, 
                                 loads_array: np.ndarray, quadrature_order: int = 2) -> np.ndarray:
    """
    Compute the equivalent nodal force vector (12,) for the element.
    """
    logger.info("Step 8: Computing force vector for element %d with length %e", element_idx, element_length)
    
    num_integration_points = max(2, quadrature_order)
    xi_points, weights = np.polynomial.legendre.leggauss(num_integration_points)
    detJ = element_length / 2
    
    x_gauss = (xi_points + 1) * detJ
    interpolated_forces = interpolate_loads(x_gauss, loads_array)
    
    N_matrix, _, _ = shape_functions(xi_points, element_length)
    
    Fe = np.zeros(12)
    for g in range(len(xi_points)):
        N_g = N_matrix[g]
        f_g = interpolated_forces[g]
        Fe += np.einsum('ij,j->i', N_g, f_g) * weights[g] * detJ
    
    logger.debug("Element force vector Fe:\n%s", Fe)
    return Fe

# --- Main Function ---
def main() -> None:
    try:
        int_logger, final_logger = configure_loggers()
        global logger
        logger = int_logger

        logger.info("Step 0: Starting Tensor Validation Test Script for Timoshenko Beam")
        
        base_dir = os.path.join(fem_model_root, "jobs", "base")
        job_dir = os.path.join(fem_model_root, "jobs", "job_0001")
        geometry_array, material_array, mesh_dictionary, load_array = load_data(base_dir, job_dir, logger)
        
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
        
        final_results_str = "\n"
        for elem_idx, result in results.items():
            conn_str = ", ".join(map(str, mesh_dictionary["connectivity"][elem_idx])) if "connectivity" in mesh_dictionary else "N/A"
            header = f"Element {elem_idx} (Length: {result['Element Length']:.6e}, Connectivity: {conn_str})\n"
            ke_table = np.array2string(result["Element Stiffness Matrix"], precision=4, suppress_small=True)
            fe_table = np.array2string(result["Element Force Vector"], precision=4, suppress_small=True)
            final_results_str += f"{header}Ke:\n{ke_table}\nFe:\n{fe_table}\n\n"
        
        final_logger.info("Final Results:\n%s", final_results_str)
        logger.info("Tensor Validation Test Completed!")
        
    except Exception as e:
        logger.error("Error in main function: %s", e, exc_info=True)
        raise

if __name__ == "__main__":
    main()
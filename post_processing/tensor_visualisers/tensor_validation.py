import os
import sys
import logging
import numpy as np
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

# --- Configure Logging ---
log_dir = os.path.join(fem_model_root, "post_processing", "tensor_validation_results", "logs")
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "tensor_validation.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path, mode="a")
    ]
)

logger = logging.getLogger(__name__)

def validate_file_path(file_path: str) -> None:
    """Validate that the file exists."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

def validate_geometry_array(geometry_array: np.ndarray) -> None:
    """Validate the geometry array shape and properties."""
    if geometry_array.shape != (1, 20):
        raise ValueError(f"Unexpected shape for geometry_array: {geometry_array.shape}. Expected (1, 20).")

def validate_material_array(material_array: np.ndarray) -> None:
    """Validate the material array shape and properties."""
    if material_array.shape != (1, 4):
        raise ValueError(f"Unexpected shape for material_array: {material_array.shape}. Expected (1, 4).")

def load_data(base_dir: str, job_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
    """Load and validate data from the base and job directories."""
    try:
        logger.info("📥 Loading base settings...")
        geometry_path = os.path.join(base_dir, "geometry.txt")
        material_path = os.path.join(base_dir, "material.txt")
        solver_path = os.path.join(base_dir, "solver.txt")
        mesh_path = os.path.join(job_dir, "mesh.txt")
        load_path = os.path.join(job_dir, "load.txt")

        validate_file_path(geometry_path)
        validate_file_path(material_path)
        validate_file_path(solver_path)
        validate_file_path(mesh_path)
        validate_file_path(load_path)

        geometry_array = parse_geometry(geometry_path)
        material_array = parse_material(material_path)
        mesh_dictionary = parse_mesh(mesh_path)
        load_array = parse_load(load_path)

        validate_geometry_array(geometry_array)
        validate_material_array(material_array)

        if load_array.shape[1] != 9:
            raise ValueError(f"Unexpected shape for load_array: {load_array.shape}. Expected (N_loads, 9).")

        logger.info("✅ Data successfully loaded.")
        return geometry_array, material_array, mesh_dictionary, load_array

    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        raise

def shape_functions(xi: np.ndarray, L: float, poly_order: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute shape functions and their derivatives for an Euler-Bernoulli beam element."""
    if poly_order != 3:
        raise ValueError("Euler-Bernoulli elements require cubic (3rd order) shape functions.")

    xi = np.atleast_1d(xi)
    n = xi.shape[0]
    detJ = L / 2

    # Axial shape functions
    N1 = 0.5 * (1 - xi)
    N4 = 0.5 * (1 + xi)
    dN1_dxi = -0.5 * np.ones_like(xi)
    dN4_dxi = 0.5 * np.ones_like(xi)
    d2N1_dxi2 = np.zeros_like(xi)
    d2N4_dxi2 = np.zeros_like(xi)

    # Bending shape functions
    N2 = 0.25 * (1 - xi)**2 * (1 + 2 * xi)
    N5 = 0.25 * (1 + xi)**2 * (1 - 2 * xi)
    dN2_dxi = 0.5 * (1 - xi) * (1 + 2 * xi) - 0.25 * (1 - xi)**2 * 2
    dN5_dxi = -0.5 * (1 + xi) * (1 - 2 * xi) + 0.25 * (1 + xi)**2 * (-2)
    d2N2_dxi2 = 1.5 * xi - 0.5
    d2N5_dxi2 = -1.5 * xi + 0.5

    # Rotational shape functions
    N3 = (1 / detJ) * dN2_dxi
    N6 = (1 / detJ) * dN5_dxi
    dN3_dxi = (1 / detJ) * d2N2_dxi2
    dN6_dxi = (1 / detJ) * d2N5_dxi2
    d2N3_dxi2 = (1 / detJ) * (1.5 * np.ones_like(xi))
    d2N6_dxi2 = (1 / detJ) * (-1.5 * np.ones_like(xi))

    # Assemble shape function matrices
    N_matrix = np.stack((
        np.column_stack((N1, np.zeros(n), np.zeros(n), N4, np.zeros(n), np.zeros(n))),
        np.column_stack((np.zeros(n), N2, np.zeros(n), np.zeros(n), N5, np.zeros(n))),
        np.column_stack((np.zeros(n), np.zeros(n), N3, np.zeros(n), np.zeros(n), N6))
    ), axis=1)

    dN_dxi_matrix = np.stack((
        np.column_stack((dN1_dxi, np.zeros(n), np.zeros(n), dN4_dxi, np.zeros(n), np.zeros(n))),
        np.column_stack((np.zeros(n), dN2_dxi, np.zeros(n), np.zeros(n), dN5_dxi, np.zeros(n))),
        np.column_stack((np.zeros(n), np.zeros(n), dN3_dxi, np.zeros(n), np.zeros(n), dN6_dxi))
    ), axis=1)

    d2N_dxi2_matrix = np.stack((
        np.column_stack((d2N1_dxi2, np.zeros(n), np.zeros(n), d2N4_dxi2, np.zeros(n), np.zeros(n))),
        np.column_stack((np.zeros(n), d2N2_dxi2, np.zeros(n), np.zeros(n), d2N5_dxi2, np.zeros(n))),
        np.column_stack((np.zeros(n), np.zeros(n), d2N3_dxi2, np.zeros(n), np.zeros(n), d2N6_dxi2))
    ), axis=1)

    return N_matrix, dN_dxi_matrix, d2N_dxi2_matrix

def compute_stiffness_matrix(element_length: float, geometry_array: np.ndarray, material_array: np.ndarray, quadrature_order: int = 3) -> np.ndarray:
    """Compute the element stiffness matrix using numerical integration."""
    A = geometry_array[0, 1]  # Cross-sectional area
    I_z = geometry_array[0, 4]  # Moment of inertia (z-axis)
    E = material_array[0, 0]  # Young’s modulus

    EA = E * A
    EI_z = E * I_z

    D = np.array([
        [EA, 0, 0],
        [0, 0, 0],
        [0, 0, EI_z]
    ])

    num_integration_points = max(2, quadrature_order)
    xi_points, weights = np.polynomial.legendre.leggauss(num_integration_points)

    _, dN_dxi_matrix, _ = shape_functions(xi_points, element_length)
    detJ = element_length / 2

    Ke = np.zeros((6, 6))
    for i in range(num_integration_points):
        Ke += dN_dxi_matrix[i].T @ D @ dN_dxi_matrix[i] * detJ * weights[i]

    return Ke

def compute_element_force_vector(element_idx: int, element_length: float, loads_array: np.ndarray, quadrature_order: int = 3) -> np.ndarray:
    """Compute the equivalent nodal force vector for an element."""
    num_integration_points = max(2, quadrature_order)
    xi_points, weights = np.polynomial.legendre.leggauss(num_integration_points)

    detJ = element_length / 2
    Fe = np.zeros(6)  # Shape: (6,)

    x_gauss = (xi_points + 1) * detJ  # Shape: (num_integration_points,)
    interpolated_forces = interpolate_loads(x_gauss, loads_array)  # Shape: (num_integration_points, 9)

    log_file = os.path.join(log_dir, f"element_{element_idx}_force_vector.log")
    with open(log_file, "w") as f:
        for i, xi in enumerate(x_gauss):
            q_xi = interpolated_forces[i]  # Shape: (9,)

            # Ensure q_xi has 9 components
            if len(q_xi) != 9:
                raise ValueError(f"Expected 9 components in q_xi, but got {len(q_xi)}.")

            # Extract forces and moments (Fx, Fy, Fz, Mx, My, Mz)
            forces_moments = q_xi[3:]  # Shape: (6,)

            # Reshape into a column vector (6, 1)
            forces_moments_reshaped = forces_moments.reshape(6, 1)  # Shape: (6, 1)

            # Compute shape functions
            N_matrix, _, _ = shape_functions(xi_points[i], element_length)  # Shape: (3, 6)

            # Perform matrix multiplication: N_matrix.T (6, 3) @ forces_moments_reshaped (6, 1)
            Fe += (N_matrix.T @ forces_moments_reshaped).flatten() * detJ * weights[i]  # Shape: (6,)

            # Logging
            f.write(f"xi = {xi:.6f}, Weight = {weights[i]:.6f}\n")
            f.write(f"Interpolated Load at xi: {np.array2string(q_xi, precision=6, separator=', ')}\n")
            f.write(f"Shape Function Matrix at xi:\n{np.array2string(N_matrix, precision=6, separator=', ')}\n\n")

    return Fe  # Shape: (6,)

def main() -> None:
    """Main function to execute the tensor validation test."""
    try:
        logger.info("🚀 Starting Tensor Validation Test Script")

        base_dir = os.path.join(fem_model_root, "jobs", "base")
        job_dir = os.path.join(fem_model_root, "jobs", "job_0001")  # Updated to point to job_0001
        output_dir = os.path.join(fem_model_root, "post_processing", "tensor_validation_results")
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        geometry_array, material_array, mesh_dictionary, load_array = load_data(base_dir, job_dir)

        logger.info("🛠 Computing Element Matrices...")
        results = {}
        for elem_idx, elem_length in enumerate(mesh_dictionary["element_lengths"]):
            Fe = compute_element_force_vector(elem_idx, elem_length, load_array)
            Ke = compute_stiffness_matrix(elem_length, geometry_array, material_array)

            results[elem_idx] = {
                "Element Force Vector": Fe.reshape(6, 1),
                "Element Stiffness Matrix": Ke
            }

        # Save results
        for elem_idx, result in results.items():
            output_file = os.path.join(output_dir, f"element_{elem_idx}_results.txt")
            with open(output_file, "w") as f:
                np.savetxt(f, result["Element Force Vector"], fmt="%.6e")
                f.write("\n")
                np.savetxt(f, result["Element Stiffness Matrix"], fmt="%.6e")

        logger.info("✅ Tensor Validation Test Completed!")

    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
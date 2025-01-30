import os
import sys
import glob
import logging
import time
import numpy as np

# Adjust Python Path to include project root
script_dir = os.path.dirname(os.path.abspath(__file__))
fem_model_root = os.path.abspath(os.path.join(script_dir, '..'))
if fem_model_root not in sys.path:
    sys.path.insert(0, fem_model_root)

# Import required modules
from pre_processing.parsing.geometry_parser import parse_geometry
from pre_processing.parsing.mesh_parser import parse_mesh
from pre_processing.parsing.material_parser import parse_material
from pre_processing.parsing.solver_parser import parse_solver
from pre_processing.parsing.load_parser import parse_load  
from processing.solver_registry import get_solver_registry
from simulation_runner.static_simulation import StaticSimulationRunner
from pre_processing.element_library.element_1D_base import Element1DBase  # Updated usage of Element1DBase

# Configure logging

log_file_path = os.path.join(script_dir, "run_job.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path, mode="a")
    ]
)
def main():
    """
    Main workflow for executing FEM simulations.
    """

    # --- 1. Initialize Job Directories ---
    jobs_dir = os.path.join(fem_model_root, 'jobs')
    results_dir = os.path.join(fem_model_root, 'post_processing', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load solver registry
    try:
        solver_registry = get_solver_registry()
    except Exception as e:
        logging.error(f"Failed to load solver registry: {e}")
        return

    # Identify available job directories
    job_dirs = [d for d in glob.glob(os.path.join(jobs_dir, 'job_*')) if os.path.isdir(d)]
    if not job_dirs:
        logging.warning("No job directories found.")
        return

    # --- 2. Iterate Over Jobs ---
    for job_dir in job_dirs:
        case_name = os.path.basename(job_dir)
        logging.info(f"Starting simulation for job: {case_name}")
        start_time = time.time()

        try:
            # --- 3. Parse Input Files ---
            geometry_array = parse_geometry(os.path.join(jobs_dir, 'base', "geometry.txt"))
            mesh_dictionary = parse_mesh(os.path.join(job_dir, "mesh.txt"))
            material_array = parse_material(os.path.join(jobs_dir, 'base', "material.txt"))
            solver_array = parse_solver(os.path.join(jobs_dir, 'base', "solver.txt"))
            load_array = parse_load(os.path.join(job_dir, "load.txt"))

            # --- 4. Select Solver ---
            solver_name = next((solver for solver in solver_array if solver.lower() != "off"), None)
            if solver_name is None or solver_name not in solver_registry:
                logging.error(f"No valid solver found for {case_name}. Skipping.")
                continue

            solver_func = solver_registry[solver_name]
            logging.info(f"Using solver: {solver_name}")

            # --- 5. Instantiate Elements Using `Element1DBase` ---
            elements_base = Element1DBase(
                geometry_array=geometry_array,
                material_array=material_array,
                mesh_dictionary=mesh_dictionary,
                load_array=load_array
            )

            # --- 6. Access Precomputed Ke and Fe ---
            element_stiffness_matrices = elements_base.element_stiffness_matrices 
            element_force_vectors = elements_base.element_force_vectors

            # --- 7. Run the Simulation ---
            runner = StaticSimulationRunner(
                settings={
                    "elements": np.array(elements_base.elements_instances),  # ✅ Ensure it's a NumPy array
                    "mesh_dictionary": mesh_dictionary,
                    "material_array": material_array,
                    "geometry_array": geometry_array,
                    "solver_name": solver_name,
                    "element_stiffness_matrices": element_stiffness_matrices,
                    "element_force_vectors": element_force_vectors,
                },
                job_name=case_name
            )

            runner.setup_simulation()
            runner.run(solver_func)  # ✅ Pass solver function explicitly
            runner.save_primary_results()

            logging.info(f"✅ Simulation completed for {case_name} in {time.time() - start_time:.2f} sec.")


        except Exception as e:
            logging.error(f"Unexpected error in job {case_name}: {e}", exc_info=True)
            continue

# Entry Point
if __name__ == "__main__":
    main()
# workflow_manager/run_job.py

import os
import sys
import glob
import logging
import time
import numpy as np

# Adjust Python Path to include the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
fem_model_root = os.path.abspath(os.path.join(script_dir, '..'))
if fem_model_root not in sys.path:
    sys.path.insert(0, fem_model_root)

# Import parsers, solvers, and elements
from workflow_manager.parsing.parser_base import ParserBase
from processing.solver_registry import get_solver_registry
from simulation_runner.static_simulation import StaticSimulationRunner
from pre_processing.element_library.euler_bernoulli import EulerBernoulliBeamElement
from pre_processing.element_library.timoshenko import TimoshenkoBeamElement

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("run_job.log", mode="a")]
)

ELEMENT_CLASS_MAP = {
    "EulerBernoulliBeamElement": EulerBernoulliBeamElement,
    "TimoshenkoBeamElement": TimoshenkoBeamElement,
}

def main():
    """Main workflow handler to parse input data and dispatch it to the correct solver."""
    
    jobs_dir = os.path.join(fem_model_root, 'jobs')
    results_dir = os.path.join(fem_model_root, 'post_processing', 'results')
    os.makedirs(results_dir, exist_ok=True)

    try:
        solver_registry = get_solver_registry()
    except Exception as e:
        logging.error(f"Failed to load solver registry: {e}")
        return

    job_dirs = [d for d in glob.glob(os.path.join(jobs_dir, 'job_*')) if os.path.isdir(d)]
    if not job_dirs:
        logging.warning("No job directories found. Ensure job_* directories are present.")
        return

    for job_dir in job_dirs:
        case_name = os.path.basename(job_dir)
        logging.info(f"Starting simulation for job: {case_name}")
        start_time = time.time()

        try:
            parser = ParserBase()

            # Parse all required input data
            geometry_data = parser.geometry_parser(os.path.join(jobs_dir, 'base', "geometry.txt"))
            mesh_data = parser.mesh_parser(os.path.join(job_dir, "mesh.txt"))
            material_props = parser.material_parser(os.path.join(jobs_dir, 'base', "material.txt"))
            solver_data = parser.solver_parser(os.path.join(jobs_dir, 'base', "solver.txt"))

            # Retrieve active solver
            solver_name = next((solver for solver, name in solver_data.items() if name.lower() != "off"), None)
            if solver_name is None or solver_name not in solver_registry:
                logging.error(f"No valid solver found for {case_name}. Skipping.")
                continue

            solver_func = solver_registry[solver_name]
            logging.info(f"Using solver: {solver_name} for {case_name}.")

            # Instantiate elements without modifying stiffness matrices
            elements_instances = []
            for element_id, connectivity in enumerate(mesh_data["connectivity"]):
                try:
                    element_class = ELEMENT_CLASS_MAP.get(geometry_data.get("element_type", "EulerBernoulliBeamElement"))

                    element = element_class(
                        element_id=element_id,
                        material_properties=material_props,
                        section_properties=geometry_data,  # Directly using parsed data
                        mesh_data=mesh_data,
                        dof_per_node=6,
                        dof_map=[0, 1, 5]
                    )

                    elements_instances.append(element)

                except Exception as e:
                    logging.error(f"Error creating element {element_id}: {e}", exc_info=True)
                    continue

            if not elements_instances:
                logging.error(f"No valid elements instantiated for {case_name}. Skipping job.")
                continue

            # Send parsed data to the correct solver
            if solver_name.startswith("Static"):
                runner = StaticSimulationRunner(
                    settings={
                        "elements": elements_instances,
                        "solver_name": solver_name
                    },
                    job_name=case_name
                )

            elif solver_name.startswith("Dynamic"):
                logging.error("Dynamic solver not implemented yet.")
                continue

            elif solver_name.startswith("Modal"):
                logging.error("Modal solver not implemented yet.")
                continue

            else:
                logging.error(f"Unknown solver type: {solver_name}. Skipping job.")
                continue

            # Dispatch to solver (run independently)
            runner.setup_simulation()
            runner.run(solver_func)
            runner.save_primary_results()

            logging.info(f"Simulation completed for {case_name} in {time.time() - start_time:.2f} seconds.")

        except Exception as e:
            logging.error(f"Unexpected error in job {case_name}: {e}", exc_info=True)
            continue

if __name__ == "__main__":
    main()
import logging
import numpy as np
import os
import datetime
from scipy.sparse import coo_matrix
from processing.assembly import assemble_global_matrices
from processing.boundary_conditions import apply_boundary_conditions
from processing.static.solver import solve_fem_system  # ✅ Importing solver function

logger = logging.getLogger(__name__)

class StaticSimulationRunner:
    """
    Handles static finite element analysis.

    Responsibilities:
        - Assembles global stiffness matrix and force vector.
        - Applies boundary conditions.
        - Solves the FEM system using a specified solver.
        - Computes nodal displacements and reaction forces.
        - Saves results with time-stamped filenames.
    """

    def __init__(self, settings, job_name):
        """
        Initializes the Static FEM Solver.

        Args:
            settings (dict): Simulation settings passed from `run_job.py`.
            job_name (str): Unique identifier for the simulation job.
        """
        self.settings = settings
        self.job_name = job_name
        self.results = {}

        # ✅ Extract and verify required settings
        self.elements = self.settings.get("elements", np.array([]))
        self.mesh_dictionary = self.settings.get("mesh_dictionary", {})

        # ✅ Corrected check for empty arrays/dictionaries
        if self.elements.size == 0 or len(self.mesh_dictionary) == 0:
            raise ValueError("❌ Error: Missing elements or mesh data in settings!")

        self.solver_name = self.settings.get("solver_name", None)
        self.element_stiffness_matrices = self.settings.get("element_stiffness_matrices", None)
        self.element_force_vectors = self.settings.get("element_force_vectors", None)

        # ✅ Create directories for storing results
        self.primary_results_dir = os.path.join("post_processing", job_name, "primary_results")
        os.makedirs(self.primary_results_dir, exist_ok=True)

    def setup_simulation(self):
        """
        ✅ Prepares the simulation environment.
        - Logs job details.
        - Ensures necessary directories exist.
        """
        logger.info(f"✅ Setting up static simulation for job: {self.job_name}...")

    def run(self, solver_func):
        """
        Executes the static FEM simulation.

        Args:
            solver_func (function): Solver function provided by `run_job.py`.
        """
        logger.info(f"Running static simulation for job: {self.job_name}...")

        try:
            # ✅ Extract total degrees of freedom (12 DOFs per node)
            num_nodes = len(self.mesh_dictionary["node_ids"])
            total_dof = num_nodes * 12

            # ✅ Convert element stiffness matrices to COO format (if needed)
            stiffness_matrices_coo = [
                Ke if isinstance(Ke, coo_matrix) else Ke.tocoo()
                for Ke in self.element_stiffness_matrices
            ]

            # ✅ Convert element force vectors to NumPy arrays (if needed)
            force_vectors_dense = [
                np.asarray(Fe) if not isinstance(Fe, np.ndarray) else Fe
                for Fe in self.element_force_vectors
            ]

            # ✅ Assemble Global Matrices
            _, _, K_global, F_global = assemble_global_matrices(
                elements=self.elements,
                element_stiffness_matrices=stiffness_matrices_coo,
                element_force_vectors=force_vectors_dense,
                total_dof=total_dof
            )

            # ✅ Apply Boundary Conditions
            K_mod, F_mod = apply_boundary_conditions(K_global, F_global)

            # ✅ Solve using external solver function
            logger.info(f"Solving FEM system using `{self.solver_name}`.")
            displacements = solver_func(K_mod, F_mod)

            # ✅ Compute Reaction Forces
            reaction_forces = K_global @ displacements

            # ✅ Store Results
            self.results = {
                "global_stiffness_matrix": K_global,
                "global_force_vector": F_global,
                "nodal_displacements": displacements,
                "reaction_forces": reaction_forces,
            }

            logger.info("✅ Static simulation completed successfully.")

        except Exception as e:
            logger.error(f"❌ Error during static analysis of {self.job_name}: {e}", exc_info=True)
            raise

    def save_primary_results(self):
        """Saves primary simulation results with timestamped filenames."""
        if not self.results:
            logger.warning(f"No results to save for job: {self.job_name}.")
            return

        logger.info("✅ Saving primary results...")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = os.path.join(self.primary_results_dir)
        os.makedirs(results_dir, exist_ok=True)

        for key_name, matrix in self.results.items():
            try:
                if isinstance(matrix, np.ndarray):
                    filename = f"{self.job_name}_{key_name}_{timestamp}.txt"
                    file_path = os.path.join(results_dir, filename)

                    with open(file_path, 'w') as f:
                        f.write("# Static Analysis Results\n")
                        f.write(f"# Job: {self.job_name}\n")
                        f.write(f"# Data Key: {key_name}\n")
                        f.write(f"# Timestamp: {timestamp}\n")
                        f.write("# Data:\n")
                        np.savetxt(f, matrix, fmt="%.6f", delimiter=",")

                    logger.info(f"✅ Saved {key_name} -> {file_path}")

            except Exception as e:
                logger.error(f"❌ Error saving {key_name}: {e}", exc_info=True)

        logger.info("✅ Primary results successfully saved.")
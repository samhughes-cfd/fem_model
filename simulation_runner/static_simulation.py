import logging
import numpy as np
import os
import datetime
from scipy.sparse.linalg import spsolve
from processing.assembly import assemble_global_matrices
from processing.boundary_conditions import apply_boundary_conditions
from processing.solver_registry import get_solver_registry


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
            settings (dict): Simulation settings from `run_job.py`, including:
                - "elements" (list): Instantiated element objects.
                - "mesh_dictionary" (dict): Contains all mesh-related NumPy arrays.
                - "material_array" (np.array): Material properties array.
                - "geometry_array" (np.array): Sectional properties array.
                - "solver_name" (str): Name of solver function.
                - "element_stiffness_matrices" (scipy.sparse.csr_matrix): Precomputed Ke.
                - "element_force_vectors" (scipy.sparse.csr_matrix): Precomputed Fe.
            job_name (str): Unique identifier for the simulation job.
        """
        self.settings = settings
        self.job_name = job_name
        self.results = {}
        self.solver_registry = get_solver_registry()

        # Create directories for storing results
        self.primary_results_dir = os.path.join("post_processing", job_name, "primary_results")
        os.makedirs(self.primary_results_dir, exist_ok=True)

    def setup_simulation(self):
        """Logs that the simulation is being set up."""
        logging.info(f"Setting up static simulation for job: {self.job_name}...")

    def run(self, solver_func):
        """
        Executes the static FEM simulation.

        Steps:
            1. Assemble global stiffness matrix and force vector.
            2. Apply boundary conditions.
            3. Solve the system using the assigned solver.
            4. Compute reaction forces.
            5. Store results.

        Args:
            solver_func (function): Solver function from the registry.

        Raises:
            RuntimeError: If there is an error in solving the FEM system.
        """
        logging.info(f"Running static simulation for job: {self.job_name}...")

        try:
            # ✅ Extract Mesh Data
            mesh_dictionary = self.settings["mesh_dictionary"]
            num_nodes = len(mesh_dictionary["node_ids"])
            total_dof = num_nodes * 12  # ✅ 12 DOFs per node

            # ✅ Assemble Global Matrices Using Precomputed Ke and Fe
            logging.info("Assembling global stiffness matrix and force vector...")
            K_global, F_global = assemble_global_matrices(
                elements=self.settings["elements"],
                stiffness_matrices=self.settings["element_stiffness_matrices"],
                force_vectors=self.settings["element_force_vectors"],
                total_dof=total_dof
            )

            # ✅ Log Matrix Dimensions
            logging.info(f"K_global shape: {K_global.shape}, F_global shape: {F_global.shape}")

            # ✅ Apply Boundary Conditions
            logging.info("Applying boundary conditions...")
            K_mod, F_mod = apply_boundary_conditions(K_global, F_global)

            # ✅ Perform Solver Diagnostics
            try:
                cond_number = np.linalg.cond(K_mod.toarray())  # Convert sparse matrix for diagnostics
                if cond_number > 1e12:  # High condition number means nearly singular
                    logging.warning(f"⚠️  Warning: K_mod is nearly singular (Condition Number: {cond_number:.2e}).")

                rank = np.linalg.matrix_rank(K_mod.toarray())
                logging.info(f"Matrix rank: {rank}/{K_mod.shape[0]}")
            except Exception as diag_error:
                logging.error(f"Error computing condition number or rank: {diag_error}", exc_info=True)

            # ✅ Solve the System Using Efficient Sparse Solver
            logging.info(f"Solving system using `{self.settings['solver_name']}` solver...")
            try:
                if isinstance(K_mod, np.ndarray):
                    displacements = solver_func(K_mod, F_mod)
                else:
                    displacements = spsolve(K_mod, F_mod)  # ✅ Use sparse solver if available
                logging.info("Solver completed successfully.")
            except np.linalg.LinAlgError as e:
                logging.error(f"Linear algebra error in solver `{self.settings['solver_name']}`: {e}", exc_info=True)
                raise RuntimeError(f"Solver `{self.settings['solver_name']}` failed due to a singular matrix.")

            # ✅ Compute Reaction Forces
            reaction_forces = K_global @ displacements  # ✅ Efficient sparse matrix multiplication

            # ✅ Vectorized Calculation of Element Displacement Vectors
            num_elements = len(self.settings["elements"])
            expected_size = num_elements * 12
            if len(displacements) != expected_size:
                raise ValueError(f"Displacement vector size mismatch: Expected {expected_size}, got {len(displacements)}")

            element_displacement_vectors = np.split(displacements, num_elements)
            logging.info("Element displacement vectors processed successfully.")

            # ✅ Store Results
            self.results = {
                "global_stiffness_matrix": K_global,
                "global_force_vector": F_global,
                "global_displacement_vector": displacements,
                "global_reaction_force_vector": reaction_forces,
                "element_stiffness_matrices": self.settings["element_stiffness_matrices"],
                "element_force_vectors": self.settings["element_force_vectors"],
                "element_displacement_vectors": element_displacement_vectors,
                "reaction_force_vectors": reaction_forces[:12],  # First 12 DOFs constrained
            }

            logging.info("Static simulation completed successfully.")

        except Exception as e:
            logging.error(f"Error during static analysis of {self.job_name}: {e}", exc_info=True)
            raise

    def save_primary_results(self):
        """
        Saves primary simulation results as text files with timestamped filenames in `post_processing/`.
        """
        if not self.results:
            logging.warning(f"No results to save for job: {self.job_name}.")
            return

        logging.info("Saving primary results...")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # ✅ Add date-time tag

        for key_name, matrix in self.results.items():
            try:
                if isinstance(matrix, np.ndarray):
                    filename = f"{self.job_name}_{key_name}_{timestamp}.txt"  # ✅ Append timestamp
                    file_path = os.path.join(self.primary_results_dir, filename)

                    with open(file_path, 'w') as f:
                        f.write("# Static Analysis Results\n")
                        f.write(f"# Job: {self.job_name}\n")
                        f.write(f"# Data Key: {key_name}\n")
                        f.write("# Data:\n")
                        np.savetxt(f, matrix, fmt="%.6f", delimiter=",")

                    logging.info(f"Saved {key_name} -> {file_path}")

            except Exception as e:
                logging.error(f"Error saving {key_name}: {e}", exc_info=True)

        logging.info("Primary results successfully saved.")
import logging
import numpy as np
import os
import datetime
from scipy.sparse import coo_matrix, csr_matrix
from processing.assembly import assemble_global_matrices
from processing.boundary_conditions import apply_boundary_conditions
from processing.static.solver import solve_fem_system

logger = logging.getLogger(__name__)

class StaticSimulationRunner:
    """
    Handles static finite element analysis.
    """

    def __init__(self, settings, job_name):
        self.settings = settings
        self.job_name = job_name
        self.results = {}

        # ✅ Extract and verify required settings
        self.elements = self.settings.get("elements", np.array([]))
        self.mesh_dictionary = self.settings.get("mesh_dictionary", {})

        if self.elements.size == 0 or not self.mesh_dictionary:
            raise ValueError("❌ Error: Missing elements or mesh data in settings!")

        self.solver_name = self.settings.get("solver_name", None)
        self.element_stiffness_matrices = self.settings.get("element_stiffness_matrices", None)
        self.element_force_vectors = self.settings.get("element_force_vectors", None)

        # ✅ Ensure stiffness matrices are in sparse format
        self.element_stiffness_matrices = self._ensure_sparse_format(self.element_stiffness_matrices)

        # ✅ Ensure force vectors are properly formatted as 1D NumPy arrays
        self.element_force_vectors = (
            np.array([np.asarray(Fe).flatten() for Fe in self.element_force_vectors], dtype=object)
            if self.element_force_vectors is not None else None
        )

        # ✅ Create directories for storing results
        self.primary_results_dir = os.path.join("post_processing", job_name, "primary_results")
        os.makedirs(self.primary_results_dir, exist_ok=True)

    def _ensure_sparse_format(self, matrices):
        """Converts all matrices in `matrices` to COO sparse format if they are not already."""
        if matrices is None:
            return None
        return np.array([
            coo_matrix(matrix) if not isinstance(matrix, coo_matrix) else matrix
            for matrix in matrices
        ], dtype=object)

    def setup_simulation(self):
        """✅ Prepares the simulation environment."""
        logger.info(f"✅ Setting up static simulation for job: {self.job_name}...")

    def run(self, solver_func):
        """Executes the static FEM simulation."""
        logger.info(f"Running static simulation for job: {self.job_name}...")

        try:
            num_nodes = len(self.mesh_dictionary["node_ids"])
            total_dof = num_nodes * 12

            # ✅ Assemble Global Matrices
            K_global, F_global = assemble_global_matrices(
                elements=self.elements,
                element_stiffness_matrices=self.element_stiffness_matrices,
                element_force_vectors=self.element_force_vectors,
                total_dof=total_dof
            )

            if K_global is None or F_global is None:
                raise ValueError("❌ Error: Global matrices could not be assembled!")

            # ✅ Ensure `F_global` is correctly formatted before modifying boundary conditions
            if F_global is not None:
                F_global = np.asarray(F_global).flatten()

            # ✅ Apply Boundary Conditions
            K_mod, F_mod = apply_boundary_conditions(K_global, F_global)

            # ✅ Solve using external solver function
            logger.info(f"Solving FEM system using `{self.solver_name}`.")
            U_global = solver_func(K_mod, F_mod)

            if U_global is None:
                raise ValueError("❌ Error: Solver returned no results!")

            # ✅ Compute Reaction Forces
            R_global = K_global @ U_global

            # ✅ Store Global Results
            self.results = {
                "global_stiffness_matrix": K_global,
                "global_force_vector": F_global,
                "global_deformation_vector": U_global,
                "global_reaction_force_vector": R_global,
            }

            # ✅ Store By-Element Results in Single Files
            self._store_by_element_results(U_global, R_global)

            logger.info("✅ Static simulation completed successfully.")

        except Exception as e:
            logger.error(f"❌ Error during static analysis of {self.job_name}: {e}", exc_info=True)
            raise

    def _store_by_element_results(self, U_global, R_global):
        """Stores by-element stiffness matrices, force vectors, displacements, and reaction forces in single files."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = self.primary_results_dir
        os.makedirs(results_dir, exist_ok=True)

        # ✅ Define filenames
        filenames = {
            "stiffness_matrix": os.path.join(results_dir, f"{self.job_name}_element_stiffness_matrices_{timestamp}.txt"),
            "force_vector": os.path.join(results_dir, f"{self.job_name}_element_force_vectors_{timestamp}.txt"),
            "deformation_vector": os.path.join(results_dir, f"{self.job_name}_element_displacements_{timestamp}.txt"),
            "reaction_force_vector": os.path.join(results_dir, f"{self.job_name}_element_reaction_forces_{timestamp}.txt"),
        }

        # ✅ Open files once and write element-wise data with headers
        with open(filenames["stiffness_matrix"], 'w') as f_Ke, \
             open(filenames["force_vector"], 'w') as f_Fe, \
             open(filenames["deformation_vector"], 'w') as f_Ue, \
             open(filenames["reaction_force_vector"], 'w') as f_Re:

            for i, element in enumerate(self.elements):
                element_id = element.element_id
                dof_indices = element.assemble_global_dof_indices(element_id)

                # Write headers
                f_Ke.write(f"\n# Element ID: {element_id}\n")
                f_Fe.write(f"\n# Element ID: {element_id}\n")
                f_Ue.write(f"\n# Element ID: {element_id}\n")
                f_Re.write(f"\n# Element ID: {element_id}\n")

                # Write element stiffness matrix
                Ke_coo = coo_matrix(self.element_stiffness_matrices[i])
                for row, col, value in zip(Ke_coo.row, Ke_coo.col, Ke_coo.data):
                    f_Ke.write(f"{row}, {col}, {value:.6f}\n")

                # Write element force vector
                np.savetxt(f_Fe, self.element_force_vectors[i].reshape(1, -1), fmt="%.6f", delimiter=",")

                # Write element displacement vector
                np.savetxt(f_Ue, U_global[dof_indices].reshape(1, -1), fmt="%.6f", delimiter=",")

                # Write element reaction force vector
                np.savetxt(f_Re, R_global[dof_indices].reshape(1, -1), fmt="%.6f", delimiter=",")

        logger.info("✅ By-element results saved successfully.")

    def save_primary_results(self):
        """Saves primary global simulation results."""
        if not self.results:
            logger.warning(f"⚠️ No results to save for job: {self.job_name}.")
            return

        logger.info("✅ Saving global results...")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = self.primary_results_dir
        os.makedirs(results_dir, exist_ok=True)

        for key_name, data in self.results.items():
            file_path = os.path.join(results_dir, f"{self.job_name}_{key_name}_{timestamp}.txt")

            try:
                with open(file_path, 'w') as f:
                    f.write("# Static Analysis Global Results\n")
                    f.write(f"# Job: {self.job_name}\n")
                    f.write(f"# Data Key: {key_name}\n")
                    f.write(f"# Timestamp: {timestamp}\n")
                    f.write("# Data:\n")

                    if isinstance(data, np.ndarray):
                        np.savetxt(f, data, fmt="%.6f", delimiter=",")
                    elif isinstance(data, (coo_matrix, csr_matrix)):
                        coo_data = coo_matrix(data)
                        for row, col, value in zip(coo_data.row, coo_data.col, coo_data.data):
                            f.write(f"{row}, {col}, {value:.6f}\n")

                logger.info(f"✅ Saved {key_name} -> {file_path}")

            except Exception as e:
                logger.error(f"❌ Error saving {key_name}: {e}", exc_info=True)

        logger.info("✅ Global results successfully saved.")
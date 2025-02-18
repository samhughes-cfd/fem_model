# simulation_runner\static_simulation.py

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

        # ✅ Store the start time when the simulation begins
        self.start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # ✅ Initialize primary and secondary results storage
        self.primary_results = {
            "global": {},  # Stores stiffness, forces, displacements, reactions
            "element": {"data": []},  # Stores element-wise results
        }
        self.secondary_results = {
            "global": {},  # Placeholder for secondary computed results (stress, strain, etc.)
            "element": {"data": []},  # Placeholder for element-wise secondary results
        }

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
            if self.element_force_vectors is not None
            else None
        )

        # ✅ Define the results directory using the simulation start time
        self.primary_results_dir = os.path.join(
            "post_processing", "results", f"{self.job_name}_{self.start_time}"
        )

    def _ensure_sparse_format(self, matrices):
        """Converts all matrices in `matrices` to COO sparse format if they are not already."""
        if matrices is None:
            return None
        return np.array([
            coo_matrix(matrix) if not isinstance(matrix, coo_matrix) else matrix
            for matrix in matrices
        ], dtype=object)

    def setup_simulation(self):
        """✅ Prepares the simulation environment (a placeholder for any setup tasks)."""
        logger.info(f"✅ Setting up static simulation for job: {self.job_name}...")

    # -------------------------------------------------------------------------
    # 1) ASSEMBLE GLOBAL MATRICES
    # -------------------------------------------------------------------------
    def assemble_global_matrices(self):
        """
        Assembles the global stiffness matrix (K_global) and
        the global force vector (F_global).
        """
        logger.info("Assembling global matrices...")
        num_nodes = len(self.mesh_dictionary["node_ids"])
        total_dof = num_nodes * 12

        K_global, F_global = assemble_global_matrices(
            elements=self.elements,
            element_stiffness_matrices=self.element_stiffness_matrices,
            element_force_vectors=self.element_force_vectors,
            total_dof=total_dof
        )

        if K_global is None or F_global is None:
            raise ValueError("❌ Error: Global matrices could not be assembled!")

        # Ensure F_global is a flattened 1D array
        F_global = np.asarray(F_global).flatten()

        return K_global, F_global

    # -------------------------------------------------------------------------
    # 2) MODIFY GLOBAL MATRICES (BOUNDARY CONDITIONS)
    # -------------------------------------------------------------------------
    def modify_global_matrices(self, K_global, F_global):
        """
        Applies boundary conditions (e.g., constraints) to the
        global stiffness matrix and force vector.
        """
        logger.info("Applying boundary conditions...")
        K_mod, F_mod = apply_boundary_conditions(K_global, F_global)
        return K_mod, F_mod

    # -------------------------------------------------------------------------
    # 3) SOLVE LINEAR SYSTEM
    # -------------------------------------------------------------------------
    def solve_linear_system(self, K_mod, F_mod):
        """
        Solves the linear system using the specified solver
        and returns the global displacement vector.
        """
        logger.info(f"Solving FEM system using `{self.solver_name}`.")
        U_global = solve_fem_system(K_mod, F_mod, self.solver_name)

        if U_global is None:
            raise ValueError("❌ Error: Solver returned no results!")

        return U_global

    # -------------------------------------------------------------------------
    # 4) COMPUTE PRIMARY RESULTS
    # -------------------------------------------------------------------------
    def compute_primary_results(self, K_global, F_global, K_mod, F_mod, U_global):
        """
        Computes key global results (reactions, etc.) from the displacement solution.
        Stores them in `self.primary_results`.
        """
        logger.info("Computing primary results...")

        # Global reaction forces
        R_global = K_global @ U_global

        # ✅ Store Global Results including modified matrices
        self.primary_results["global"] = {
            "stiffness_matrix": K_global,  # Original
            "force_vector": F_global,      # Original
            "modified_stiffness_matrix": K_mod,  # ✅ Added modified matrix
            "modified_force_vector": F_mod,      # ✅ Added modified force vector
            "deformation_vector": U_global,
            "reaction_force_vector": R_global,
        }

        # Compute and Store Element-wise Results
        self.primary_results["element"]["data"] = []

        for element_id, element_matrix in enumerate(self.element_stiffness_matrices):
            force_vector = self.element_force_vectors[element_id]
            deformation_vector = U_global[:len(force_vector)]  # Extract relevant part of U
            reaction_force_vector = element_matrix @ deformation_vector

            self.primary_results["element"]["data"].append({
                "element_id": element_id,
                "stiffness_matrix": element_matrix,
                "force_vector": force_vector,
                "deformation_vector": deformation_vector,
                "reaction_force_vector": reaction_force_vector,
            })

        logger.info(f"Primary results computed. Element count: {len(self.primary_results['element']['data'])}")


    # -------------------------------------------------------------------------
    # 5) SAVE PRIMARY RESULTS (GLOBAL + ELEMENT)
    # -------------------------------------------------------------------------
    def save_primary_results(self, output_dir=None):
        """
        Saves both global and element-wise primary simulation results.
        """
        if not self.primary_results.get("global") and not self.primary_results.get("element", {}).get("data"):
            logger.warning(f"⚠️ No primary results to save for job: {self.job_name}.")
            return

        logger.info("✅ Saving primary results...")

        # Use simulation start time for consistent timestamps
        timestamp = self.start_time

        # Use the passed-in output directory (top-level job folder) if provided;
        # the runner will create its own "primary_results" subfolder.
        results_dir = output_dir if output_dir else self.primary_results_dir
        primary_results_dir = os.path.join(results_dir, "primary_results")
        os.makedirs(primary_results_dir, exist_ok=True)

        # -------------------------------------------------------------------------
        # Helper: Write a uniform signature header into a file.
        # -------------------------------------------------------------------------
        def _write_signature_header(f, scale, key_name, timestamp):
            f.write(f"# Static simulation\n")
            f.write(f"# Job: {self.job_name}\n")
            f.write(f"# Scale: {scale}\n")
            f.write(f"# Data key: {key_name}\n")
            f.write(f"# Timestamp (runner starts): {timestamp}\n")
            f.write("# Data:\n")

        # -------------------------------------------------------------------------
        # 5.1) SAVE GLOBAL PRIMARY RESULTS
        # -------------------------------------------------------------------------
        logger.info("Saving global primary results...")

        global_scale = "global"  # Derived from the dictionary key
        global_results = self.primary_results.get(global_scale, {})

        for key_name, data in global_results.items():
            file_path = os.path.join(primary_results_dir,
                                    f"{self.job_name}_static_{global_scale}_{key_name}_{timestamp}.txt")
            try:
                with open(file_path, "w") as f:
                    _write_signature_header(f, global_scale, key_name, timestamp)
                    if isinstance(data, np.ndarray):
                        np.savetxt(f, data, fmt="%.6f", delimiter=",")
                    elif isinstance(data, (coo_matrix, csr_matrix)):
                        coo_data = coo_matrix(data)
                        for row, col, value in zip(coo_data.row, coo_data.col, coo_data.data):
                            f.write(f"{row}, {col}, {value:.6f}\n")
                logger.info(f"✅ Saved global {key_name} -> {file_path}")
            except Exception as e:
                logger.error(f"❌ Error saving global result '{key_name}': {e}", exc_info=True)

        # -------------------------------------------------------------------------
        # 5.2) SAVE ELEMENT-WISE PRIMARY RESULTS
        # -------------------------------------------------------------------------
        logger.info("Saving element-wise primary results...")

        element_scale = "element"  # Derived from the dictionary key
        element_data = self.primary_results.get(element_scale, {}).get("data", [])
        if not element_data:
            logger.info("⚠️ No element data found. Skipping element-level save.")
            return

        # Assume all element dictionaries share the same keys; get them from the first element.
        # Exclude 'element_id' from the result keys.
        result_keys = [key for key in element_data[0].keys() if key != "element_id"]

        # Prepare filenames for each result type using the unified naming format.
        filenames = {
            key: os.path.join(primary_results_dir,
                            f"{self.job_name}_static_{element_scale}_{key}_{timestamp}.txt")
            for key in result_keys
        }

        # Use ExitStack to manage multiple file handles.
        from contextlib import ExitStack
        with ExitStack() as stack:
            file_handles = {
                key: stack.enter_context(open(filenames[key], "w"))
                for key in result_keys
            }
            # Write a signature header at the top of each element result file.
            for key in result_keys:
                _write_signature_header(file_handles[key], element_scale, key, timestamp)
    
            # Write data for each element into the appropriate file.
            for elem_info in element_data:
                element_id = elem_info["element_id"]
                for key in result_keys:
                    file_handles[key].write(f"\n# Element ID: {element_id}\n")
                    value = elem_info[key]
                    # Handle sparse matrix values (e.g., stiffness matrices)
                    if hasattr(value, "tocoo"):
                        coo_val = value.tocoo()
                        for row, col, val in zip(coo_val.row, coo_val.col, coo_val.data):
                            file_handles[key].write(f"{row}, {col}, {val:.6f}\n")
                    else:
                        # Assume it's a NumPy array
                        np.savetxt(file_handles[key], value.reshape(1, -1), fmt="%.6f", delimiter=",")
    
        # Log saved files for element-level results.
        for key in result_keys:
            logger.info(f"✅ Saved element {key} -> {filenames[key]}")
        logger.info("✅ Element-level primary results saved successfully.")

    # -------------------------------------------------------------------------
    # 6) COMPUTE SECONDARY RESULTS (PLACEHOLDERS)
    # -------------------------------------------------------------------------

    def compute_secondary_results(self):
        """
        Computes secondary results such as stress, strain, and energy.
        These are derived from primary results and stored in `self.secondary_results`.
        """
        logger.info("Computing secondary results...")

        if not self.primary_results["global"]:
            logger.warning("⚠️ Cannot compute secondary results: No primary results available.")
            return

        # Example: Compute stress & strain (Placeholder)
        self.secondary_results["global"]["stress"] = np.array([0.0])  # Replace with actual stress computation
        self.secondary_results["global"]["strain"] = np.array([0.0])  # Replace with actual strain computation

        # Example: Compute element-wise secondary results (Placeholder)
        self.secondary_results["element"]["data"] = []

        for element in self.primary_results["element"]["data"]:
            self.secondary_results["element"]["data"].append({
                "element_id": element["element_id"],
                "stress": np.array([0.0]),  # Replace with real calculations
                "strain": np.array([0.0]),  # Replace with real calculations
            })

        logger.info(f"Secondary results computed for {len(self.secondary_results['element']['data'])} elements.")

    # -------------------------------------------------------------------------
    # 7) SAVE SECONDARY RESULTS (PLACEHOLDERS)
    # -------------------------------------------------------------------------

    def save_secondary_results(self, output_dir=None):
        """
        Saves secondary simulation results (e.g., stress, strain, energy).
        These results are derived from primary results.

        Parameters
        ----------
        output_dir : str, optional
            If provided, saves the secondary results to this directory
            instead of the default `self.primary_results_dir`.
        """
        if not self.secondary_results.get("global") and not self.secondary_results.get("element", {}).get("data"):
            logger.warning(f"⚠️ No secondary results to save for job: {self.job_name}.")
            return

        logger.info("Saving secondary results...")

        # Use the simulation start time for consistent timestamps
        timestamp = self.start_time

        # Ensure the correct results directory
        results_dir = output_dir if output_dir else self.primary_results_dir
        secondary_results_dir = os.path.join(results_dir, "secondary_results")
        os.makedirs(secondary_results_dir, exist_ok=True)

        # -------------------------------------------------------------------------
        # 7.1) SAVE GLOBAL SECONDARY RESULTS
        # -------------------------------------------------------------------------
        logger.info("Saving global secondary results...")

        for key_name, data in self.secondary_results.get("global", {}).items():
            file_path = os.path.join(secondary_results_dir, f"{self.job_name}_{key_name}_{timestamp}.txt")

            try:
                with open(file_path, "w") as f:
                    f.write("# Static simulation\n")
                    f.write(f"# Job: {self.job_name}\n")
                    f.write(f"# Data key: {key_name}\n")
                    f.write(f"# Timestamp (runner start): {timestamp}\n")
                    f.write("# Data:\n")

                    if isinstance(data, np.ndarray):
                        np.savetxt(f, data, fmt="%.6f", delimiter=",")
                    elif isinstance(data, (coo_matrix, csr_matrix)):
                        coo_data = coo_matrix(data)
                        for row, col, value in zip(coo_data.row, coo_data.col, coo_data.data):
                            f.write(f"{row}, {col}, {value:.6f}\n")

                logger.info(f"Saved global secondary result '{key_name}' -> {file_path}")

            except Exception as e:
                logger.error(f"Error saving global secondary result '{key_name}': {e}", exc_info=True)

        # -------------------------------------------------------------------------
        # 5.2) SAVE ELEMENT-WISE SECONDARY RESULTS
        # -------------------------------------------------------------------------
        logger.info("Saving element-wise secondary results...")

        element_data = self.secondary_results.get("element", {}).get("data", [])
        if not element_data:
            logger.info("⚠️ No element secondary data found. Skipping element-level save.")
            return

        filenames = {
            "stress": os.path.join(secondary_results_dir, f"{self.job_name}_element_stress_{timestamp}.txt"),
            "strain": os.path.join(secondary_results_dir, f"{self.job_name}_element_strain_{timestamp}.txt"),
            "energy": os.path.join(secondary_results_dir, f"{self.job_name}_element_energy_{timestamp}.txt"),
        }

        with open(filenames["stress"], 'w') as f_stress, \
            open(filenames["strain"], 'w') as f_strain, \
            open(filenames["energy"], 'w') as f_energy:

            for elem_info in element_data:
                element_id = elem_info["element_id"]

                # Write headers
                f_stress.write(f"\n# Element ID: {element_id}\n")
                f_strain.write(f"\n# Element ID: {element_id}\n")
                f_energy.write(f"\n# Element ID: {element_id}\n")

                # 1) Stress
                if "stress" in elem_info:
                    np.savetxt(f_stress, elem_info["stress"].reshape(1, -1), fmt="%.6f", delimiter=",")

                # 2) Strain
                if "strain" in elem_info:
                    np.savetxt(f_strain, elem_info["strain"].reshape(1, -1), fmt="%.6f", delimiter=",")

                # 3) Energy (if computed)
                if "energy" in elem_info:
                    np.savetxt(f_energy, elem_info["energy"].reshape(1, -1), fmt="%.6f", delimiter=",")

        logger.info(f"Saved element stress -> {filenames['stress']}")
        logger.info(f"Saved element strain -> {filenames['strain']}")
        logger.info(f"Saved element energy -> {filenames['energy']}")

        logger.info("Element-level secondary results saved successfully.")

    # -------------------------------------------------------------------------
    # MAIN ENTRY POINT
    # -------------------------------------------------------------------------
    def run(self):
        """
        Orchestrates the static FEM simulation in multiple steps,
        to allow granular performance measurements.
        """
        logger.info(f"Running static simulation for job: {self.job_name}...")
        self.setup_simulation()

        try:
            # 1) Assemble; assemble global matrices from element-wise contributions
            K_global, F_global = self.assemble_global_matrices()

            # 2) Modify; apply boundary conditions
            K_mod, F_mod = self.modify_global_matrices(K_global, F_global)

            # 3) Solve; solve the F = K U for U
            U_global = self.solve_linear_system(K_mod, F_mod)

            # 4) Compute primary results
            self.compute_primary_results(K_global, F_global, K_mod, F_mod, U_global)

            # 5) Save primary results
            self.save_primary_results()

            # 6) Placeholder for advanced post-processing
            # self.compute_secondary_results()

            # 7) Save secondary results
            # self.save_secondary_results()

            logger.info("Static simulation completed successfully.")

        except Exception as e:
            logger.error(f"Error during static analysis of {self.job_name}: {e}", exc_info=True)
            raise
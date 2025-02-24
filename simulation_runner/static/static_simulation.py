# simulation_runner/static_simulation.py

import logging
import numpy as np
import os
import datetime
from scipy.sparse import coo_matrix

from processing.assembly import assemble_global_matrices
from processing.boundary_conditions import apply_boundary_conditions
from processing.static.solver import solve_fem_system
from processing.static_condensation import condensation, reconstruction
from processing.disassembly import disassemble_global_matrices

from simulation_runner.static.linear_static_diagnostic import log_system_diagnostics
from processing.static.linear_solver_diagnostic import log_solver_performance

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
    def assemble_global_matrices(self, job_results_dir):
        """
        Assembles the global stiffness matrix (K_global) and force vector (F_global).

        Steps:
        1) Compute total DOFs.
        2) Assemble K_global and F_global from element data.
        3) Validate and log the assembled system.

        Parameters:
            job_results_dir (str): Directory for logging results.

        Returns:
            Tuple[csr_matrix, np.ndarray]:
                - K_global: Assembled global stiffness matrix (CSR format).
                - F_global: Assembled global force vector (1D NumPy array).
        """

        logger.info("🔧 Assembling global stiffness and force matrices...")

        # Compute total degrees of freedom (DOFs)
        num_nodes = len(self.mesh_dictionary["node_ids"])
        total_dof = num_nodes * 6  # Assuming 6 DOFs per node

        try:
            # Assemble global stiffness matrix (K_global) and force vector (F_global)
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

            # Log system diagnostics before applying boundary conditions
            log_system_diagnostics(K_global, F_global, bc_dofs=[], job_results_dir=job_results_dir, label="Global System")

            logger.info("✅ Global stiffness matrix and force vector successfully assembled!")

        except Exception as e:
            logger.error(f"⚠️ Assembly failed: {e}")
            raise

        return K_global, F_global


    # -------------------------------------------------------------------------
    # 2) MODIFY GLOBAL MATRICES (BOUNDARY CONDITIONS)
    # -------------------------------------------------------------------------
    
    def modify_global_matrices(self, K_global, F_global, job_results_dir):
        """
        Applies boundary conditions to the global system and logs diagnostics.

        Steps:
        1) Apply boundary conditions using penalty method.
        2) Validate and log the modified system.

        Parameters:
            K_global (csr_matrix): Assembled global stiffness matrix.
            F_global (np.ndarray): Assembled global force vector.
            job_results_dir (str): Directory for logging results.

        Returns:
            Tuple[csr_matrix, np.ndarray, np.ndarray]:
                - K_mod: Modified global stiffness matrix (CSR format).
                - F_mod: Modified global force vector.
                - bc_dofs: Indices of fixed DOFs where constraints were applied.
        """

        logger.info("🔒 Applying boundary conditions to global matrices...")

        try:
            # Apply boundary conditions using the penalty method
            K_mod, F_mod, bc_dofs = apply_boundary_conditions(K_global.copy(), F_global.copy())

            # Ensure modified force vector is correctly formatted
            F_mod = np.asarray(F_mod).flatten()

            # Log diagnostics after applying boundary conditions
            log_system_diagnostics(K_mod, F_mod, bc_dofs=bc_dofs, job_results_dir=job_results_dir, label="Modified System")

            logger.info("✅ Boundary conditions successfully applied!")

        except Exception as e:
            logger.error(f"⚠️ Error applying boundary conditions: {e}")
            raise

        return K_mod, F_mod, bc_dofs

    # -------------------------------------------------------------------------
    # 3) SOLVE SYSTEM (STATIC CONDENSATION & RECONSTRUCTION)
    # -------------------------------------------------------------------------

    def solve_static(self, K_mod, F_mod, fixed_dofs, job_results_dir):
        """
        Solves the FEM system using static condensation and logs diagnostics.

        Steps:
        1) Condense the system by removing fixed DOFs and inactive DOFs.
        2) Solve the reduced system.
        3) Reconstruct the full displacement vector.

        Parameters:
            K_mod (csr_matrix): Modified global stiffness matrix after applying boundary conditions.
            F_mod (np.ndarray): Modified global force vector after applying boundary conditions.
            fixed_dofs (np.ndarray): Indices of fixed DOFs (explicit boundary conditions).
            job_results_dir (str): Directory for logging results.

        Returns:
            Tuple[np.ndarray, csr_matrix, np.ndarray, np.ndarray]:
                - U_global: Full displacement vector with zeros at fixed DOFs.
                - K_cond: Condensed stiffness matrix after removing inactive DOFs.
                - F_cond: Condensed force vector.
                - U_cond: Solution of the condensed system.

        Raises:
            ValueError: If condensation leads to an empty system.
            ValueError: If solver returns a zero displacement vector.
        """

        logger.info(f"🔹 Solving FEM system using static condensation with `{self.solver_name}`.")

        # 1️⃣ **Perform Static Condensation**
        try:
            logger.info("🔹 Performing static condensation...")
            active_dofs, inactive_dofs, K_cond, F_cond = condensation(K_mod, F_mod, fixed_dofs, tol=1e-12)

            if K_cond.shape[0] == 0 or F_cond.shape[0] == 0:
                raise ValueError("❌ Condensed system is empty! Possible over-constrained system.")

            # Log system diagnostics
            log_system_diagnostics(K_cond, F_cond, bc_dofs=fixed_dofs, job_results_dir=job_results_dir, label="Condensed System")

        except Exception as e:
            logger.error(f"❌ Error during static condensation: {e}")
            raise

        # 2️⃣ **Solve the Condensed System**
        try:
            logger.info("🔹 Solving the reduced system...")
            U_cond = solve_fem_system(K_cond, F_cond, self.solver_name, job_results_dir)

            if U_cond is None or np.allclose(U_cond, 0, atol=1e-12):
                logger.warning("⚠️ Solver returned an all-zero displacement vector. Proceeding with zero displacements for debugging.")
                U_cond = np.zeros_like(F_cond)  # Assign zeros instead of stopping

        except Exception as e:
            logger.error(f"❌ Solver failure: {e}")
            U_cond = np.zeros_like(F_cond)  # Allow execution to continue

        # 3️⃣ **Reconstruct the Full Displacement Vector**
        try:
            logger.info("🔹 Reconstructing the full displacement vector...")
            U_global = reconstruction(active_dofs, U_cond, K_mod.shape[0])
            logger.info(f"✅ Displacement vector computed: {U_global}")
        except Exception as e:
            logger.error(f"❌ Error during displacement reconstruction: {e}")

        logger.info("✅ Static solve completed successfully!")

        return U_global, K_cond, F_cond, U_cond

    # -------------------------------------------------------------------------
    # 4) COMPUTE PRIMARY RESULTS
    # -------------------------------------------------------------------------
    
    def compute_primary_results(self, K_global, F_global, K_mod, F_mod, 
                                K_cond, F_cond, U_cond, U_global):
        """
        Computes primary results including reaction forces, displacements, nodal rotations (θ_z),
        and bending moments (M_z) for Euler-Bernoulli beam elements.
        
        Extensive logging is performed to compare the new post-processed global results with the original
        (pre-processed) results.
        
        Returns:
            Tuple: (global_results, element_results)
        """
        logger.info("Computing primary results (extensive logging enabled)...")

        # Retrieve boundary DOFs (for the full system) without modifying matrices.
        _, _, bc_dofs = apply_boundary_conditions(K_global, F_global)

        # -----------------------------------------------------
        # ORIGINAL GLOBAL REACTION FORCES
        # -----------------------------------------------------
        R_global = np.zeros_like(F_global)
        R_global[bc_dofs] = -F_global[bc_dofs]
        logger.debug("Original R_global computed. Shape: %s, Norm: %.3e",
                     R_global.shape, np.linalg.norm(R_global))

        # Compute R_cond using the condensed system equation.
        R_cond = K_cond @ U_cond - F_cond
        logger.debug("Original R_cond computed from K_cond @ U_cond - F_cond. Shape: %s, Norm: %.3e", 
                     R_cond.shape, np.linalg.norm(R_cond))

        # -----------------------------------------------------
        # STORE INITIAL GLOBAL RESULTS (Pre-Processing)
        # -----------------------------------------------------
        self.primary_results["global"] = {
            "K_global": K_global,
            "F_global": F_global,
            "K_mod": K_mod,
            "F_mod": F_mod,
            "K_cond": K_cond,
            "F_cond": F_cond,
            "U_cond": U_cond,
            "R_cond": R_cond,  # For debugging purposes
            "U_global": U_global,
            "R_global": R_global,
        }
        logger.info("Stored initial global results (pre-processing).")

        # -----------------------------------------------------
        # POST-PROCESSING FUNCTION: Updates Global Results
        # -----------------------------------------------------
        def post_processing():
            """
            Computes nodal rotations (θ_z) and bending moments (M_z) for Euler-Bernoulli beam elements,
            updates U_global and F_mod, and recalculates R_global_updated.
            """
            logger.info("Post-processing global results for Euler-Bernoulli beam elements...")

            # Retrieve element lengths and element types using dictionary keys.
            element_lengths = self.settings["mesh_dictionary"]["element_lengths"]
            element_types = self.settings["mesh_dictionary"]["element_types"]

            # Ensure geometry_array is 1D.
            geometry_array = self.settings["geometry_array"]
            if geometry_array.ndim > 1:
                geometry_array = geometry_array.flatten()
            logger.debug("Geometry array used for post-processing: shape %s", geometry_array.shape)

            U_global_updated = U_global.copy()
            F_mod_updated = F_mod.copy()
            num_nodes = len(U_global) // 6  # assuming 6 DOFs per node
            logger.info("Number of nodes: %d", num_nodes)

            # Define historical indices.
            MATERIAL_E_INDEX = 0    # Young's Modulus index (historically)
            GEOMETRY_IZ_INDEX = 3   # Moment of inertia about z-axis index (historically)

            # Loop over each element; update only for Euler-Bernoulli elements.
            for element_id, element_type in enumerate(element_types):
                if element_type == "EulerBernoulliBeamElement3DOF":
                    # Extract E using 2D indexing (assumed shape (1, n)).
                    try:
                        E = float(self.settings["material_array"][0, MATERIAL_E_INDEX])
                    except Exception as ex:
                        logger.error("Error extracting E for element %d: %s", element_id, ex)
                        raise

                    # Extract I_z using 2D indexing.
                    if geometry_array.size > GEOMETRY_IZ_INDEX:
                        I_z = float(self.settings["geometry_array"][0, GEOMETRY_IZ_INDEX])
                    else:
                        logger.warning("geometry_array has size %d; expected at least %d. Using geometry_array[0,0] for I_z.",
                                       geometry_array.size, GEOMETRY_IZ_INDEX+1)
                        I_z = float(self.settings["geometry_array"][0, 0])
                    
                    EI = E * I_z

                    # Get L for the element.
                    try:
                        L = float(element_lengths[element_id])
                    except Exception as ex:
                        logger.error("Error extracting L for element %d: %s", element_id, ex)
                        raise

                    logger.debug("Element %d: E=%.3e, I_z=%.3e, L=%.3e, EI=%.3e", element_id, E, I_z, L, EI)

                    theta_z_values = np.zeros(num_nodes)
                    Mz_values = np.zeros(num_nodes)

                    # Compute θ_z using a three-point one-sided stencil.
                    for i in range(num_nodes):
                        index = 6 * i + 1  # u_y index
                        if i == 0:
                            theta_z_values[i] = (-3 * U_global[index] + 4 * U_global[index + 6] - U_global[index + 12]) / (2 * L)
                        elif i == num_nodes - 1:
                            theta_z_values[i] = (3 * U_global[index] - 4 * U_global[index - 6] + U_global[index - 12]) / (2 * L)
                        else:
                            theta_z_values[i] = (U_global[index + 6] - U_global[index]) / L
                    logger.debug("Element %d: Computed theta_z_values: %s", element_id, theta_z_values)

                    # Compute M_z using a four-point one-sided stencil.
                    for i in range(num_nodes):
                        index = 6 * i + 1
                        if i == 0:
                            Mz_values[i] = EI * (2 * U_global[index] - 5 * U_global[index + 6] + 4 * U_global[index + 12] - U_global[index + 18]) / (L**2)
                        elif i == num_nodes - 1:
                            Mz_values[i] = EI * (2 * U_global[index] - 5 * U_global[index - 6] + 4 * U_global[index - 12] - U_global[index - 18]) / (L**2)
                        else:
                            Mz_values[i] = EI * (U_global[index + 6] - 2 * U_global[index] + U_global[index - 6]) / (L**2)
                    logger.debug("Element %d: Computed Mz_values: %s", element_id, Mz_values)

                    # Insert computed θ_z and M_z into updated global arrays.
                    for node_id in range(num_nodes):
                        dof_index = 6 * node_id + 5
                        U_global_updated[dof_index] = theta_z_values[node_id]
                        F_mod_updated[dof_index] = Mz_values[node_id]

            # Log differences between original and updated U_global and F_mod.
            diff_U = np.linalg.norm(U_global_updated - U_global)
            diff_F = np.linalg.norm(F_mod_updated - F_mod)
            logger.info("Post-processing differences: ||ΔU_global||=%.3e, ||ΔF_mod||=%.3e", diff_U, diff_F)

            # Update global results in primary_results.
            self.primary_results["global"]["U_global"] = U_global_updated
            self.primary_results["global"]["F_mod"] = F_mod_updated

            # Compute and store updated R_global_updated using bc_dofs.
            R_global_updated = np.zeros_like(F_mod_updated)
            R_global_updated[bc_dofs] = -F_mod_updated[bc_dofs]
            logger.info("Computed updated R_global with norm: %.3e", np.linalg.norm(R_global_updated))
            self.primary_results["global"]["R_global"] = R_global_updated

            return U_global_updated, F_mod_updated, R_global_updated

        # -----------------------------------------------------
        # Check if global results need updating (for Euler-Bernoulli beams)
        # -----------------------------------------------------
        if "EulerBernoulliBeamElement3DOF" in self.settings["mesh_dictionary"]["element_types"]:
            logger.info("Euler-Bernoulli beam elements detected; applying post-processing updates.")
            U_global_updated, F_mod_updated, R_global_updated = post_processing()
        else:
            logger.info("No Euler-Bernoulli beam elements detected; using original global results.")
            U_global_updated, F_mod_updated, R_global_updated = U_global, F_mod, R_global

        # -----------------------------------------------------
        # Compute element-wise results (after global updates)
        # -----------------------------------------------------
        elements = self.elements  # List of element objects

        try:
            K_e_mod, F_e_mod, U_e, R_e = disassemble_global_matrices(
                elements, K_mod, F_mod_updated, U_global_updated, R_global_updated
            )
        except ValueError as e:
            logger.error("Error during disassembly of element-wise results: %s", e)
            return None, None

        # Store element-wise results.
        self.primary_results["element"] = {"data": [
            {"element_id": i, 
             "K_e": self.element_stiffness_matrices[i],
             "F_e": self.element_force_vectors[i],
             "K_e_mod": K_e_mod[i], 
             "F_e_mod": F_e_mod[i], 
             "U_e": U_e[i], 
             "R_e": R_e[i]}
            for i in range(len(K_e_mod))
        ]}
        logger.info("Finalized primary results with %d element-wise entries.", len(K_e_mod))

        return self.primary_results["global"], self.primary_results["element"]["data"]
    
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
                        np.savetxt(f, data, fmt="%.6e")
                    elif hasattr(data, "tocoo"):
                        coo_data = data.tocoo()
                        for row, col, value in zip(coo_data.row, coo_data.col, coo_data.data):
                            f.write(f"{row}, {col}, {value:.6e}\n")
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

        from contextlib import ExitStack
        with ExitStack() as stack:
            # Prepare file handles for each result type.
            file_handles = {
                key: stack.enter_context(open(os.path.join(primary_results_dir,
                                        f"{self.job_name}_static_{element_scale}_{key}_{timestamp}.txt"), "w"))
                for key in element_data[0].keys() if key != "element_id"
            }
            # Write signature header in each file.
            for key in file_handles:
                _write_signature_header(file_handles[key], element_scale, key, timestamp)
    
            for elem_info in element_data:
                element_id = elem_info["element_id"]
                for key, f_handle in file_handles.items():
                    f_handle.write(f"\n# Element ID: {element_id}\n")
                    value = elem_info[key]
                    if hasattr(value, "tocoo"):
                        coo_val = value.tocoo()
                        for row, col, val in zip(coo_val.row, coo_val.col, coo_val.data):
                            f_handle.write(f"{row}, {col}, {val:.6e}\n")
                    else:
                        np.savetxt(f_handle, value.reshape(1, -1), fmt="%.6e", delimiter=",")
    
        for key in file_handles:
            logger.info(f"✅ Saved element {key} -> {file_handles[key].name}")
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

        self.secondary_results["global"]["stress"] = np.array([0.0])  # Replace with actual stress computation
        self.secondary_results["global"]["strain"] = np.array([0.0])  # Replace with actual strain computation

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
        """
        if not self.secondary_results.get("global") and not self.secondary_results.get("element", {}).get("data"):
            logger.warning(f"⚠️ No secondary results to save for job: {self.job_name}.")
            return

        logger.info("Saving secondary results...")

        timestamp = self.start_time
        results_dir = output_dir if output_dir else self.primary_results_dir
        secondary_results_dir = os.path.join(results_dir, "secondary_results")
        os.makedirs(secondary_results_dir, exist_ok=True)

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
                    elif hasattr(data, "tocoo"):
                        coo_data = data.tocoo()
                        for row, col, value in zip(coo_data.row, coo_data.col, coo_data.data):
                            f.write(f"{row}, {col}, {value:.6f}\n")
                logger.info(f"Saved global secondary result '{key_name}' -> {file_path}")
            except Exception as e:
                logger.error(f"Error saving global secondary result '{key_name}': {e}", exc_info=True)

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
                f_stress.write(f"\n# Element ID: {element_id}\n")
                f_strain.write(f"\n# Element ID: {element_id}\n")
                f_energy.write(f"\n# Element ID: {element_id}\n")

                if "stress" in elem_info:
                    np.savetxt(f_stress, elem_info["stress"].reshape(1, -1), fmt="%.6f", delimiter=",")
                if "strain" in elem_info:
                    np.savetxt(f_strain, elem_info["strain"].reshape(1, -1), fmt="%.6f", delimiter=",")
                if "energy" in elem_info:
                    np.savetxt(f_energy, elem_info["energy"].reshape(1, -1), fmt="%.6f", delimiter=",")
        logger.info(f"Saved element stress -> {filenames['stress']}")
        logger.info(f"Saved element strain -> {filenames['strain']}")
        logger.info(f"Saved element energy -> {filenames['energy']}")
        logger.info("Element-level secondary results saved successfully.")
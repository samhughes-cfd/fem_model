# simulation_runner\static_simulation.py

import logging
import numpy as np
import os
from processing.assembly import assemble_global_matrices
from processing.boundary_conditions import apply_boundary_conditions as apply_bcs
from processing.solver_registry import get_solver_registry

class StaticSimulationRunner:
    """
    Handles static finite element analysis.

    Responsibilities:
        - Assembles global stiffness matrix and force vector.
        - Applies nodal loads and boundary conditions.
        - Solves the FEM system using a specified solver.
        - Computes nodal displacements and reaction forces.
        - Saves results in an organized manner.
    """

    def __init__(self, settings, job_name):
        """
        Initializes the Static FEM Solver.

        Args:
            settings (dict): 
                - "elements" (list): Instantiated element objects.
                - "stiffness_matrices" (dict): {element_id → Ke matrices} (precomputed in run_job.py).
                - "node_positions" (np.array): Actual node positions.
                - "material_props" (dict): Material properties (E, G, nu, rho).
                - "cross_section_props" (dict): Sectional properties (A, Iz, Iy, etc.).
                - "boundary_conditions" (np.array): Nodal constraints (num_nodes × 6).
                - "nodal_loads" (np.array): Applied loads (num_nodes × 6).
                - "solver_name" (str): Name of solver function.
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
            2. Apply nodal loads.
            3. Apply boundary conditions.
            4. Solve the system using the assigned solver.
            5. Compute reaction forces.
            6. Store results.
        
        Args:
            solver_func (function): Solver function from the registry.

        Raises:
            RuntimeError: If there is an error in solving the FEM system.
        """
        logging.info(f"Running static simulation for job: {self.job_name}...")

        try:
            # ✅ Extract required FEM settings
            elements = self.settings["elements"]
            stiffness_matrices = self.settings["stiffness_matrices"]
            num_nodes = len(self.settings["node_positions"])
            total_dof = num_nodes * 6  # 6 DOFs per node
            
            logging.info("Assembling global stiffness matrix and force vector...")

            # ✅ Assemble Global Matrices
            global_matrices = assemble_global_matrices(elements, stiffness_matrices, total_dof)
            K_global, F_global = global_matrices["K_global"], global_matrices["F_global"]

            # ✅ Apply Nodal Loads
            logging.info("Applying nodal loads...")
            F_global = self._apply_loads(F_global)

            # ✅ Apply Boundary Conditions
            logging.info("Applying boundary conditions...")
            constrained_dofs = self._generate_constrained_dofs(num_nodes)
            K_mod, F_mod = apply_bcs(K_global, F_global, constrained_dofs)

            # ✅ Solve the System
            logging.info(f"Solving system using `{self.settings['solver_name']}` solver...")
            try:
                displacements = solver_func(K_mod, F_mod)
            except Exception as e:
                raise RuntimeError(f"Solver `{self.settings['solver_name']}` encountered an error: {e}")

            # ✅ Compute Reaction Forces
            reaction_forces = K_global @ displacements

            # ✅ Store Results
            self.results = {
                "nodal_displacements": displacements,
                "nodal_forces": F_global,
                "reaction_forces": reaction_forces,
                "reaction_forces_at_constraints": reaction_forces[constrained_dofs],
                "elemental_stiffness_matrices": stiffness_matrices,
                "constrained_dofs": constrained_dofs,
            }

            logging.info("Static simulation completed successfully.")

        except Exception as e:
            logging.error(f"Error during static analysis of {self.job_name}: {e}", exc_info=True)
            raise

    def _apply_loads(self, F_global):
        """
        Applies nodal loads to the global force vector.

        Args:
            F_global (np.array): Initial global force vector.

        Returns:
            np.array: Updated force vector including applied loads.
        """
        loads_array = self.settings.get("nodal_loads", np.zeros_like(F_global))
        F_global += loads_array.flatten()  # Convert to 1D for summation
        return F_global

    def _generate_constrained_dofs(self, num_nodes):
        """
        Identifies constrained degrees of freedom based on boundary conditions.

        Args:
            num_nodes (int): Number of nodes in the system.

        Returns:
            list: Indices of constrained DOFs.
        """
        bc_array = self.settings.get("boundary_conditions", np.zeros((num_nodes, 6)))
        constrained_dofs = np.where(bc_array.flatten() == 1)[0]  # Convert to 1D indices

        logging.info(f"Identified constrained DOFs: {constrained_dofs.tolist()}")
        return constrained_dofs

    def save_primary_results(self):
        """
        Saves primary simulation results as text files in `post_processing/`.
        """
        if not self.results:
            logging.warning(f"No results to save for job: {self.job_name}.")
            return

        for key_name, matrix in self.results.items():
            if isinstance(matrix, np.ndarray) and matrix.size > 0:
                filename = f"{key_name}.txt"
                file_path = os.path.join(self.primary_results_dir, filename)

                with open(file_path, 'w') as f:
                    f.write("# Static Analysis Results\n")
                    f.write(f"# Job: {self.job_name}\n")
                    f.write(f"# Data Key: {key_name}\n")
                    f.write("# Data:\n")
                    np.savetxt(f, matrix, fmt="%.6f", delimiter=",")

                logging.info(f"Saved {key_name} -> {file_path}")
            else:
                logging.info(f"Skipping {key_name} (empty or non-array).")
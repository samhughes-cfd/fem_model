import logging
import numpy as np
import os
import datetime
from scipy.sparse import coo_matrix, linalg

# Import FEM processing functions (similar to static simulation)
from processing.modal.assembly import assemble_global_matrices
from processing.modal.boundary_conditions import apply_boundary_conditions
from simulation_runner.modal.modal_diagnostic import log_modal_diagnostics

logger = logging.getLogger(__name__)

class ModalSimulationRunner:
    """
    Handles modal finite element analysis (natural frequencies & mode shapes).
    """

    def __init__(self, settings, job_name):
        self.settings = settings
        self.job_name = job_name

        # ✅ Store the start time when the simulation begins
        self.start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # ✅ Initialize result storage
        self.primary_results = {"global": {}, "element": {"data": []}}
        self.secondary_results = {"global": {}, "element": {"data": []}}

        # ✅ Extract required settings
        self.elements = self.settings.get("elements", np.array([]))
        self.mesh_dictionary = self.settings.get("mesh_dictionary", {})

        if self.elements.size == 0 or not self.mesh_dictionary:
            logger.error("❌ Error: Missing elements or mesh data in settings!")
            raise ValueError("❌ Error: Missing elements or mesh data in settings!")

        self.solver_name = self.settings.get("solver_name", "eigen")
        self.element_stiffness_matrices = self._ensure_sparse_format(
            self.settings.get("element_stiffness_matrices", None)
        )
        self.element_mass_matrices = self._ensure_sparse_format(
            self.settings.get("element_mass_matrices", None)
        )

        # ✅ Define results directory
        self.primary_results_dir = os.path.join(
            "post_processing", "results", f"{self.job_name}_{self.start_time}"
        )

    def _ensure_sparse_format(self, matrices):
        """Converts matrices to sparse COO format if needed."""
        if matrices is None:
            return None
        return np.array([
            coo_matrix(matrix) if not isinstance(matrix, coo_matrix) else matrix
            for matrix in matrices
        ], dtype=object)

    def setup_simulation(self):
        """✅ Prepares the simulation environment."""
        logger.info(f"✅ Setting up modal simulation for job: {self.job_name}...")

    # -------------------------------------------------------------------------
    # 1) ASSEMBLE GLOBAL MATRICES
    # -------------------------------------------------------------------------

    def assemble_global_matrices(self, job_results_dir):
        """Assembles the global stiffness and mass matrices."""
        logger.info("🔧 Assembling global stiffness and mass matrices...")

        num_nodes = len(self.mesh_dictionary["node_ids"])
        total_dof = num_nodes * 6  # Assuming 6 DOFs per node

        try:
            K_global, M_global = assemble_global_matrices(
                elements=self.elements,
                element_stiffness_matrices=self.element_stiffness_matrices,
                element_mass_matrices=self.element_mass_matrices,
                total_dof=total_dof,
                job_results_dir=job_results_dir
            )

            if K_global is None or M_global is None:
                logger.error("❌ Error: Global matrices could not be assembled!")
                raise ValueError("❌ Error: Global matrices could not be assembled!")

            log_modal_diagnostics(K_global, M_global, job_results_dir)

            logger.info("✅ Global stiffness and mass matrices successfully assembled!")
        except Exception as e:
            logger.error(f"⚠️ Assembly failed: {e}")
            raise

        return K_global, M_global

    # -------------------------------------------------------------------------
    # 2) APPLY BOUNDARY CONDITIONS
    # -------------------------------------------------------------------------

    def modify_global_matrices(self, K_global, M_global, job_results_dir):
        """Applies boundary conditions to the modal system."""
        logger.info("🔒 Applying boundary conditions to global matrices...")

        try:
            K_mod, M_mod, bc_dofs = apply_boundary_conditions(K_global, M_global)

            log_modal_diagnostics(K_mod, M_mod, job_results_dir)

            logger.info("✅ Boundary conditions successfully applied!")
        except Exception as e:
            logger.error(f"⚠️ Error applying boundary conditions: {e}")
            raise

        return K_mod, M_mod, bc_dofs

    # -------------------------------------------------------------------------
    # 3) SOLVE MODAL SYSTEM
    # -------------------------------------------------------------------------

    def solve_modal(self, K_mod, M_mod, num_modes, job_results_dir):
        """
        Solves the modal system for natural frequencies and mode shapes.

        Parameters:
            K_mod (csr_matrix): Modified stiffness matrix.
            M_mod (csr_matrix): Modified mass matrix.
            num_modes (int): Number of modes to compute.
            job_results_dir (str): Directory for logging results.

        Returns:
            frequencies (np.ndarray): Natural frequencies (Hz).
            mode_shapes (np.ndarray): Mode shape vectors.
        """
        logger.info(f"🔹 Solving for {num_modes} natural frequencies and mode shapes...")

        try:
            eigenvalues, eigenvectors = linalg.eigsh(K_mod, k=num_modes, M=M_mod, which="SM")

            frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
            mode_shapes = eigenvectors

            logger.info(f"✅ Computed {num_modes} natural frequencies.")

            return frequencies, mode_shapes
        except Exception as e:
            logger.error(f"❌ Modal solver failure: {e}")
            raise

    # -------------------------------------------------------------------------
    # 4) SAVE PRIMARY RESULTS
    # -------------------------------------------------------------------------

    def save_primary_results(self, frequencies, mode_shapes):
        """Saves natural frequencies and mode shapes to files."""
        results_dir = os.path.join(self.primary_results_dir, "modal_results")
        os.makedirs(results_dir, exist_ok=True)

        np.savetxt(os.path.join(results_dir, f"{self.job_name}_frequencies.txt"), frequencies, fmt="%.6f")
        np.savetxt(os.path.join(results_dir, f"{self.job_name}_mode_shapes.txt"), mode_shapes, fmt="%.6f")

        logger.info("✅ Saved modal results.")

    # -------------------------------------------------------------------------
    # 5) COMPUTE SECONDARY RESULTS (PLACEHOLDERS)
    # -------------------------------------------------------------------------

    def compute_secondary_results(self, frequencies, mode_shapes):
        """Computes modal participation factors (as a placeholder)."""
        self.secondary_results["global"]["modal_participation"] = np.array([0.0])  # Placeholder

        logger.info("✅ Computed secondary modal results.")

    # -------------------------------------------------------------------------
    # 6) SAVE SECONDARY RESULTS
    # -------------------------------------------------------------------------

    def save_secondary_results(self):
        """Saves secondary modal results."""
        results_dir = os.path.join(self.primary_results_dir, "modal_results")
        os.makedirs(results_dir, exist_ok=True)

        np.savetxt(os.path.join(results_dir, f"{self.job_name}_modal_participation.txt"), 
                   self.secondary_results["global"]["modal_participation"], fmt="%.6f")

        logger.info("✅ Saved secondary modal results.")
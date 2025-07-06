# simulation_runner/static_simulation.py

import logging
import numpy as np
import os
from pathlib import Path
import datetime

from processing_OOP.static.diagnostics.linear_static_diagnostic import DiagnoseLinearStaticSystem
from processing_OOP.static.diagnostics.runtime_monitor_telemetry import RuntimeMonitorTelemetry

from processing_OOP.static.operations.preparation import PrepareLocalSystem
from processing_OOP.static.operations.assembly import AssembleGlobalSystem
from processing_OOP.static.operations.modification import ModifyGlobalSystem
from processing_OOP.static.operations.condensation import CondenseModifiedSystem

from processing_OOP.static.operations.solver import SolveCondensedSystem

from processing_OOP.static.operations.reconstruction import ReconstructGlobalSystem
from processing_OOP.static.primary_results.compute_primary_results import ComputePrimaryResults
from processing_OOP.static.secondary_results.compute_secondary_results import ComputeSecondaryResults
from processing_OOP.static.operations.disassembly import DisassembleGlobalSystem

# Configure module-level logger
logger = logging.getLogger(__name__)


class StaticSimulationRunner:
    """
    Handles static finite element analysis.
    """

    def __init__(
        self,
        elements,
        element_dictionary,
        grid_dictionary,
        element_stiffness_matrices,
        element_force_vectors,
        job_name,
        job_results_dir
    ):
        self.elements = elements
        self.element_dictionary = element_dictionary
        self.grid_dictionary = grid_dictionary
        self.element_stiffness_matrices = element_stiffness_matrices
        self.element_force_vectors = element_force_vectors
        self.job_name = job_name

        self.results_root = job_results_dir  # Use exact path passed from run_job.py
        self.primary_results_dir = os.path.join(self.results_root, "primary_results")
        self.secondary_results_dir = os.path.join(self.results_root, "secondary_results")
        self.maps_dir = os.path.join(self.results_root, "maps")
        self.logs_dir = os.path.join(self.results_root, "logs")
        self.diagnostics_dir = os.path.join(self.results_root, "diagnostics")

        self.monitor = RuntimeMonitorTelemetry(job_results_dir=self.primary_results_dir)

        # Initialize results storage
        self.primary_results = {
            "global": {},           # Global primary_results etc.
            "element": {}           # Element primary_results etc.
        }
        self.secondary_results = {
            "global": {},           # Global secondary_results etc.
            "element": {}           # Element secondary_results etc.
        }

        # Validate mesh and element data
        if len(self.elements) == 0 or not self.element_dictionary or not self.grid_dictionary:
            logger.error("❌ Error: Missing elements or mesh data in constructor!")
            raise ValueError("❌ Error: Missing elements or mesh data!")

        # General simulation settings (optional or future expansion)
        self.solver_name = None

        # Initialize intermediate system storage
        self.K_global = None
        self.F_global = None
        self.K_mod = None
        self.F_mod = None
        self.fixed_dofs = None
        self.condensed_dofs = None
        self.inactive_dofs = None
        self.K_cond = None
        self.F_cond = None
        self.U_cond = None
        self.U_global = None

    def setup_simulation(self):
        """Creates the necessary output directory structure."""
        logger.info(f"✅ Setting up static simulation for job: {self.job_name}...")
        os.makedirs(self.primary_results_dir, exist_ok=True)
        os.makedirs(self.secondary_results_dir, exist_ok=True)
        os.makedirs(self.maps_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.diagnostics_dir, exist_ok=True)
        logger.info(f"📁 Results will be saved to: {self.results_root}")

    # -------------------------------------------------------------------------
    # 0) PREPARE LOCAL SYSTEM
    # -------------------------------------------------------------------------

    def prepare_local_system(self, job_results_dir: str):
        """
        Validate and format local element stiffness matrices (Ke) and force vectors (Fe)
        using the PrepareLocalSystem helper.

        Parameters
        ----------
        job_results_dir : str
            Directory for logging local system processing.

        Returns
        -------
        element_stiffness_matrices : List[scipy.sparse.coo_matrix]
        element_force_vectors : List[numpy.ndarray]
        """
        logger.info("📦 Preparing local element stiffness matrices and force vectors...")

        preparer = PrepareLocalSystem(
            Ke_raw=self.element_stiffness_matrices,
            Fe_raw=self.element_force_vectors,
            job_results_dir=job_results_dir
        )

        try:
            Ke_formatted, Fe_formatted = preparer.validate_and_format()
        except Exception as exc:
            logger.error("❌ Local system preparation failed – see prepare_local_system.log for details")
            raise

        self.element_stiffness_matrices = Ke_formatted
        self.element_force_vectors = Fe_formatted

        logger.info("✅ Local element systems validated and formatted")
        return Ke_formatted, Fe_formatted

    # -------------------------------------------------------------------------
    # 1) ASSEMBLE GLOBAL SYSTEM
    # -------------------------------------------------------------------------

    def assemble_global_system(self, job_results_dir: str):
        """
        Assemble the global stiffness matrix (K_global) and force vector (F_global)
        using the high-performance AssembleGlobalSystem helper.

        Parameters
        ----------
        job_results_dir : str
            Directory where log files and any diagnostics should be written.

        Returns
        -------
        K_global : scipy.sparse.csr_matrix
        F_global : numpy.ndarray
            Flattened 1-D force vector.
        local_global_dof_map : list of np.ndarray
            Per-element mapping from local to global DOFs.
        """
        logger.info("🔧 Assembling global stiffness and force matrices...")

        # 1. Work out the total number of DOFs for the model
        node_ids = self.grid_dictionary.get("nodes", {}).get("ids", None)
        if node_ids is None:
            raise KeyError("Missing 'nodes' → 'ids' in grid_dictionary.")
        num_nodes = len(node_ids)
        total_dof = num_nodes * 6

        # 2. Instantiate the high-performance assembler and run it
        assembler = AssembleGlobalSystem(
            elements                   = self.elements,
            element_stiffness_matrices = self.element_stiffness_matrices,
            element_force_vectors      = self.element_force_vectors,
            total_dof                  = total_dof,
            job_results_dir            = job_results_dir
        )

        try:
            K_global, F_global, local_global_dof_map = assembler.assemble()
        except Exception as exc:
            logger.error("❌ Assembly failed – see assembly.log for details")
            raise

        # 3. Assemble global system diagnostics
        F_global = np.asarray(F_global).ravel()

        logger.info("✅ Global stiffness matrix and force vector successfully assembled")
        return K_global, F_global, local_global_dof_map

    # -------------------------------------------------------------------------
    # 2) MODIFY GLOBAL MATRICES (BOUNDARY CONDITIONS)
    # -------------------------------------------------------------------------
    def modify_global_system(
            self,
            K_global,
            F_global,
            local_global_dof_map,
            job_results_dir: str,
            *,
            fixed_dofs: np.ndarray | None = None):
        """
        Apply boundary conditions to the global system.

        Returns
        -------
        K_mod : csr_matrix
        F_mod : np.ndarray
        bc_dofs : np.ndarray
        """
        logger.info("🔒 Applying boundary conditions to global matrices…")

        # 1. Instantiate the boundary-condition helper
        modifier = ModifyGlobalSystem(
            K_global         = K_global,
            F_global         = np.asarray(F_global).ravel(),
            job_results_dir  = job_results_dir,
            fixed_dofs       = fixed_dofs,
            local_global_dof_map = local_global_dof_map   # ← pass map once
        )

        # 2. Run the BC pipeline (no args now)
        try:
            K_mod, F_mod, fixed_dofs = modifier.apply_boundary_conditions()
        except Exception:
            logger.error("❌ Boundary-condition step failed – see ModifyGlobalSystem.log")
            raise

        logger.info("✅ Boundary conditions successfully applied")
        return K_mod, F_mod, fixed_dofs

    # ----------------------------------------=---------------------------------
    # 3) CONDENSE MODIFIED SYSTEM
    # -------------------------------------------------------------------------

    def condense_modified_system(
            self,
            K_mod,
            F_mod,
            fixed_dofs,
            local_global_dof_map,
            job_results_dir: str,
            *,
            base_tol: float = 1e-12
        ):
        """
        Perform static condensation on the boundary-conditioned system using the
        high-performance `CondenseModifiedSystem` helper.

        Parameters
        ----------
        K_mod : scipy.sparse.csr_matrix
            Global stiffness matrix *after* boundary conditions have been applied.
        F_mod : np.ndarray
            Corresponding force vector (flattened to 1-D internally).
        fixed_dofs : np.ndarray
            Array of fixed DOF indices (passed straight through to the helper).
        job_results_dir : str
            Directory where condensation logs/diagnostics should be written.
        base_tol : float, optional
            Baseline tolerance; the helper auto-scales from this value.

        Returns
        -------
        condensed_dofs : np.ndarray
            DOF indices retained in the condensed system (w.r.t. the **original**
            global numbering).
        inactive_dofs : np.ndarray
            Active DOFs that were fully eliminated by the secondary condensation.
        K_cond : scipy.sparse.csr_matrix
            Condensed stiffness matrix.
        F_cond : np.ndarray
            Condensed force vector (1-D).
        """
        logger.info("📉 Condensing modified system…")

        
        # 1.  Instantiate the condensation helper
        
        condenser = CondenseModifiedSystem(
            K_mod        = K_mod,
            F_mod        = np.asarray(F_mod).ravel(),   # ensure 1-D
            fixed_dofs   = fixed_dofs,
            local_global_dof_map=local_global_dof_map,
            job_results_dir = job_results_dir,
            base_tol        = base_tol
        )

        # 2.  Execute the condensation pipeline
        
        try:
            condensed_dofs, inactive_dofs, K_cond, F_cond = (
                condenser.apply_condensation()
            )
        except Exception as exc:                       # helper already logs detail
            logger.error("❌ Condensation failed – see condensation.log for details")
            raise                                       # propagate to caller

        
        # 3.  High-level diagnostics (optional, keeps your existing style)

        logger.info(
            "✅ Static condensation complete "
            f"({len(condensed_dofs)} DOFs retained, "
            f"{len(inactive_dofs)} pruned)"
        )
        return condensed_dofs, inactive_dofs, K_cond, F_cond
    
    # -------------------------------------------------------------------------
    # 4) SOLVE CONDENSED SYSTEM
    # -------------------------------------------------------------------------

    def solve_condensed_system(self,
                            K_cond,
                            F_cond,
                            job_results_dir: str,
                            *,
                            solver_name: str = "cg",
                            preconditioner: str | None = "auto"):
        """
        Solve the condensed linear system K_cond · U_cond = F_cond.

        Parameters
        ----------
        K_cond, F_cond : csr_matrix, np.ndarray
            Condensed stiffness matrix and force vector.
        job_results_dir : str
            Folder for logs and CSV output.
        solver_name : {"cg","gmres","bicgstab","direct",...}, optional
            Which registered solver to use (default "cg").
        preconditioner : {"auto","ilu","jacobi",None}, optional
            Preconditioning strategy (default "auto").

        Returns
        -------
        U_cond : np.ndarray
            Displacements in condensed system ordering.
        """
        logger.info("🧮 Solving condensed system…")

        cond_solver = SolveCondensedSystem(
            K_cond=K_cond,
            F_cond=F_cond,
            solver_name=solver_name,
            job_results_dir=job_results_dir,
            preconditioner=preconditioner
        )

        U_cond = cond_solver.solve()
        if U_cond is None:
            raise RuntimeError(
                "Condensed solver failed — see SolveCondensedSystem.log")

        logger.info("✅ Condensed system successfully solved")
        return U_cond

    # ---------------------------------------------------------------------------------
    # 5) RECONSTRUCT GLOBAL SYSTEM
    # ---------------------------------------------------------------------------------

    def reconstruct_global_system(self,
                                    condensed_dofs: np.ndarray,
                                    U_cond: np.ndarray,
                                    total_dof: int,
                                    job_results_dir: str | Path,
                                    *,
                                    fixed_dofs: np.ndarray | None = None,
                                    inactive_dofs: np.ndarray | None = None,
                                    local_global_dof_map,) -> np.ndarray:
        """
        Reconstruct the full displacement vector from the condensed solution.

        Parameters
        ----------
        condensed_dofs : np.ndarray
            Active DOF indices in original numbering.
        U_cond : np.ndarray
            Displacements solved in condensed ordering.
        total_dof : int
            Total DOFs in the original system.
        job_results_dir : str | Path
            Directory for reconstruction logs and CSV output.
        fixed_dofs : np.ndarray | None, optional
            Fixed DOF indices for validation.
        inactive_dofs : np.ndarray | None, optional
            Inactive DOFs removed during condensation.

        Returns
        -------
        np.ndarray
            Reconstructed global displacement vector (
            ).
        """
        logger.info("🔄 Reconstructing global displacement vector…")

        reconstructor = ReconstructGlobalSystem(
            active_dofs=condensed_dofs,
            U_cond=U_cond,
            total_dofs=total_dof,
            job_results_dir=Path(job_results_dir),
            fixed_dofs=fixed_dofs,
            inactive_dofs=inactive_dofs,
            local_global_dof_map = local_global_dof_map
        )

        try:
            U_global = reconstructor.reconstruct()
        except Exception:
            logger.error("❌ Reconstruction failed – see ReconstructModifiedSystem.log")
            raise

        logger.info(
            f"✅ Displacement reconstruction complete "
            f"(min={U_global.min():.3e}, max={U_global.max():.3e})"
        )
        return U_global
    
    # ──────────────────────────────────────────────────────────────────────────
    # 6) PRIMARY RESULTS PIPELINE
    # ──────────────────────────────────────────────────────────────────────────
    def compute_primary_results(self) -> None:
        """
        1. Build *all* global-level first-order outputs (incl. reactions).
        2. Disassemble U_global & R_global back onto each element.
        3. Cache the two result sets in-memory.
        (All CSVs are persisted by the helpers themselves.)
        """
        logger.info("📊 Computing primary results …")

        # 1 ── global-level ----------------------------------------------------
        global_set = ComputePrimaryResults(
            K_global             = self.K_global,
            F_global             = self.F_global,
            K_mod                = self.K_mod,
            F_mod                = self.F_mod,
            K_cond               = self.K_cond,
            F_cond               = self.F_cond,
            U_cond               = self.U_cond,
            U_global             = self.U_global,
            local_global_dof_map = self.local_global_dof_map,
            fixed_dofs           = self.fixed_dofs,
            job_results_dir      = self.primary_results_dir,
        ).compute()

        # 2 ── element-level ---------------------------------------------------
        U_e, R_e = DisassembleGlobalSystem(
            elements             = self.elements,
            K_mod                = self.K_mod,                 # shape check only
            F_mod                = self.F_mod,
            U_global             = self.U_global,
            R_global             = global_set.R_global,
            local_global_dof_map = self.local_global_dof_map,
            element_K_raw        = self.element_stiffness_matrices,
            element_F_raw        = self.element_force_vectors,
            job_results_dir      = self.primary_results_dir,
        ).disassemble()

        # 3 ── keep results in RAM --------------------------------------------
        self.primary_results = {
            "global":  global_set,          # PrimaryResultSet object
            "element": (U_e, R_e),          # two lists from disassembly
        }
        logger.info("✅ Primary results computed and written")

    # -------------------------------------------------------------------------
    # 7) SECONDARY RESULTS PIPELINE
    # -------------------------------------------------------------------------
    def compute_secondary_results(self):
        """Compute and save secondary results (stresses, strains, etc.)"""
        #if not self.primary_results:
            #logger.error("❌ Compute primary results first!")
            #raise RuntimeError("Primary results not available")
        
        #logger.info("📈 Computing secondary results...")
        
        # 1. Compute secondary results
        #computer = ComputeSecondaryResults(
            #primary_results=self.primary_results["global"],
            #mesh_dict=self.mesh_dictionary,
            #elements=self.elements
        #)
        #secondary_set = computer.compute()
        
        # 2. Save results
        #SaveSecondaryResults(
            #job_name=self.job_name,
            #start_timestamp=self.start_time,
            #output_root=self.secondary_results_dir,
            #secondary_set=secondary_set
        #).write_all()
        
        # 3. Store in memory
        #self.secondary_results = secondary_set
        #logger.info("✅ Secondary results computed and saved")

    # -------------------------------------------------------------------------
    # TOP-LEVEL SIMULATION FLOW
    # -------------------------------------------------------------------------
    def run(self) -> None:
        """Execute the complete linear-static workflow."""
        try:
            # initialise once – writes machine header
            self.monitor = RuntimeMonitorTelemetry(job_results_dir=self.diagnostics_dir)

            # 0) housekeeping & local systems
            self.setup_simulation()                       # create folders
            self.start_time = datetime.datetime.now()

            # -----------------------------------------------------------------
            # 0  Prepare local element data
            # -----------------------------------------------------------------
            with self.monitor.stage("PrepareLocalSystem"):
                self.prepare_local_system(job_results_dir=self.primary_results_dir)

            # -----------------------------------------------------------------
            # 1  Global assembly
            # -----------------------------------------------------------------
            with self.monitor.stage("AssembleGlobalSystem"):
                (self.K_global,
                 self.F_global,
                 self.local_global_dof_map) = self.assemble_global_system(self.primary_results_dir)

            DiagnoseLinearStaticSystem(stage="Global System",
                                       A_current       = self.K_global,  
                                       b_current       = self.F_global,
                                       A_full          = self.K_global,  
                                       b_full          = self.F_global,
                                       fixed_dofs      = [], 
                                       condensed_dofs  = [],
                                       job_results_dir = self.primary_results_dir,)

            # -----------------------------------------------------------------
            # 2  Boundary conditions
            # -----------------------------------------------------------------
            with self.monitor.stage("ModifyGlobalSystem"):
                (self.K_mod,
                 self.F_mod,
                 self.fixed_dofs) = self.modify_global_system(self.K_global,
                                                              self.F_global,
                                                              self.local_global_dof_map,
                                                              self.primary_results_dir)

            DiagnoseLinearStaticSystem(stage="Modified System",
                                       A_current       = self.K_mod,
                                       b_current       = self.F_mod,
                                       A_full          = self.K_global,
                                       b_full          = self.F_global,
                                       fixed_dofs      = self.fixed_dofs,
                                       condensed_dofs  = [],
                                       job_results_dir = self.primary_results_dir,)

            # -----------------------------------------------------------------
            # 3  Static condensation
            # -----------------------------------------------------------------
            with self.monitor.stage("CondenseModifiedSystem"):
                (self.condensed_dofs,
                 self.inactive_dofs,
                 self.K_cond,
                 self.F_cond) = self.condense_modified_system(self.K_mod, 
                                                              self.F_mod,
                                                              self.fixed_dofs,
                                                              self.local_global_dof_map,
                                                              self.primary_results_dir)

            DiagnoseLinearStaticSystem(stage="Condensed System",
                                       A_current       = self.K_cond,
                                       b_current       = self.F_cond,
                                       A_full          = self.K_global,
                                       b_full          = self.F_global,
                                       fixed_dofs      = self.fixed_dofs,
                                       condensed_dofs  = self.condensed_dofs,
                                       job_results_dir = self.primary_results_dir,)

            # -----------------------------------------------------------------
            # 4  Solve condensed system
            # -----------------------------------------------------------------
            with self.monitor.stage("SolveCondensedSystem"):
                self.U_cond = self.solve_condensed_system(self.K_cond, 
                                                          self.F_cond,
                                                          self.primary_results_dir,
                                                          solver_name="cg", 
                                                          preconditioner="auto")

            # -----------------------------------------------------------------
            # 5  Reconstruct full-length displacement vector
            # -----------------------------------------------------------------
            with self.monitor.stage("ReconstructGlobalSystem"):
                total_dofs = len(self.mesh_dictionary["node_ids"]) * 6
                self.U_global = self.reconstruct_global_system(self.condensed_dofs,
                                                               self.U_cond,total_dofs,
                                                               self.primary_results_dir,
                                                               fixed_dofs=self.fixed_dofs,
                                                               inactive_dofs=self.inactive_dofs,
                                                               local_global_dof_map=self.local_global_dof_map)

            # -----------------------------------------------------------------
            # 6  Primary results (includes disassembly & CSV export)
            # -----------------------------------------------------------------
            with self.monitor.stage("ComputePrimaryResults"):
                self.compute_primary_results()

            # -----------------------------------------------------------------
            # 7  Secondary (derived) results
            # -----------------------------------------------------------------
            # with self.monitor.stage("ComputeSecondaryResults"):
            #     self.compute_secondary_results()

            logger.info("🏁 Simulation completed successfully → %s", self.results_root)

        except Exception as exc:
            logger.exception("💥 Simulation failed with critical error")
            raise RuntimeError("Simulation aborted") from exc
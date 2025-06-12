# simulation_runner/static_simulation.py

import logging
import numpy as np
import os
import datetime
from scipy.sparse import coo_matrix
import json
from pathlib import Path

from processing_OOP.static.operations.preparation import PrepareLocalSystem
from processing_OOP.static.operations.assembly import AssembleGlobalSystem
from processing_OOP.static.operations.boundary_conditions import ModifyGlobalSystem

from processing_OOP.static.operations.condensation import CondenseGlobalSystem
from processing_OOP.static.operations.solver import SolveCondensedSystem
from processing_OOP.static.operations.reconstruction import ReconstructGlobalSystem

from processing_OOP.static.operations.disassembly import DisassembleGlobalSystem
from processing_OOP.static.primary_results.compute_primary_results import ComputePrimaryResults
from processing_OOP.static.secondary_results.compute_secondary_results import ComputeSecondaryResults
from processing_OOP.static.primary_results.save_primary_results import SavePrimaryResults
from processing_OOP.static.secondary_results.save_secondary_results import SaveSecondaryResults

# diagnostic logging
from simulation_runner.static.linear_static_diagnostic import log_system_diagnostics

# Configure logging for this module
logger = logging.getLogger(__name__)

class StaticSimulationRunner:
    """
    Handles static finite element analysis.
    """

    def __init__(self, settings, job_name):
        self.settings = settings
        self.job_name = job_name

        # Store the start time when the simulation begins
        self.start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Initialize primary and secondary results storage
        self.primary_results = {
            "global": {},  # Stores stiffness, forces, displacements, reactions
            "element": {"data": []},  # Stores element-wise results
        }
        self.secondary_results = {
            "global": {},  # Placeholder for secondary computed results (stress, strain, etc.)
            "element": {"data": []},  # Placeholder for element-wise secondary results
        }

        # Extract and verify required settings
        self.elements = self.settings.get("elements", [])  # Default to empty list
        self.mesh_dictionary = self.settings.get("mesh_dictionary", {})

        # Use len() for list type check
        if len(self.elements) == 0 or not self.mesh_dictionary:
            logger.error("❌ Error: Missing elements or mesh data in settings!")
            raise ValueError("❌ Error: Missing elements or mesh data in settings!")

        # Rest of initialization code remains the same...
        self.solver_name = self.settings.get("solver_name", None)
        self.element_stiffness_matrices = self.settings.get("element_stiffness_matrices", None)
        self.element_force_vectors = self.settings.get("element_force_vectors", None)

        # Ensure stiffness matrices are in sparse format
        self.element_stiffness_matrices = self._ensure_sparse_format(self.element_stiffness_matrices)

        # Ensure force vectors are properly formatted as 1D NumPy arrays
        self.element_force_vectors = (
            np.array([np.asarray(Fe).flatten() for Fe in self.element_force_vectors], dtype=object)
            if self.element_force_vectors is not None
            else None
        )

        # Create comprehensive results directory structure
        self.results_root = os.path.join(
            "post_processing", "results", f"{self.job_name}_{self.start_time}"
        )
        self.primary_results_dir = os.path.join(self.results_root, "primary")
        self.secondary_results_dir = os.path.join(self.results_root, "secondary")
        
        # Initialize storage for intermediate systems
        self.K_global = None
        self.F_global = None
        self.K_mod = None
        self.F_mod = None
        self.bc_dofs = None
        self.condensed_dofs = None
        self.inactive_dofs = None
        self.K_cond = None
        self.F_cond = None
        self.U_cond = None
        self.U_global = None

    def _ensure_sparse_format(self, matrices):
        """Converts all matrices in `matrices` to COO sparse format if they are not already."""
        if matrices is None:
            return None
        return np.array([
            coo_matrix(matrix) if not isinstance(matrix, coo_matrix) else matrix
            for matrix in matrices
        ], dtype=object)

    def setup_simulation(self):
        """Creates results directory structure."""
        logger.info(f"✅ Setting up static simulation for job: {self.job_name}...")
        os.makedirs(self.primary_results_dir, exist_ok=True)
        os.makedirs(self.secondary_results_dir, exist_ok=True)
        logger.info(f"📁 Results will be saved to: {self.results_root}")

    # -------------------------------------------------------------------------
    # 0) PREPARE LOCAL SYSTEM
    # -------------------------------------------------------------------------

    def prepare_local_system(self, job_results_dir: str):
        """
        Validate and format all local element stiffness matrices (Ke) and force vectors (Fe)
        using the PrepareLocalSystem helper.

        Parameters
        ----------
        job_results_dir : str
            Directory where log files and diagnostics should be written.

        Returns
        -------
        element_stiffness_matrices : List[scipy.sparse.coo_matrix]
        element_force_vectors : List[numpy.ndarray]
            Validated and formatted versions of raw Ke and Fe for global assembly.
        """
        logger.info("📦 Preparing local element stiffness matrices and force vectors...")

        # 1. Instantiate validator
        preparer = PrepareLocalSystem(
            Ke_raw          = self.element_stiffness_matrices,
            Fe_raw          = self.element_force_vectors,
            job_results_dir = job_results_dir
        )

        # 2. Validate and format
        try:
            Ke_formatted, Fe_formatted = preparer.validate_and_format()
        except Exception as exc:
            logger.error("❌ Local system preparation failed – see prepare_local_system.log for details")
            raise  # re-raise to propagate failure

        # 3. Store the formatted versions
        self.element_stiffness_matrices = Ke_formatted
        self.element_force_vectors      = Fe_formatted

        logger.info("✅ Local element systems validated and formatted")
        return Ke_formatted, Fe_formatted

    # -------------------------------------------------------------------------
    # 1) ASSEMBLE GLOBAL SYSTEM
    # -------------------------------------------------------------------------

    def assemble_global_system(self, job_results_dir: str, *, parallel_assembly: bool = False):
        """
        Assemble the global stiffness matrix (K_global) and force vector (F_global)
        using the high-performance AssembleGlobalSystem helper.

        Parameters
        ----------
        job_results_dir : str
            Directory where log files and any diagnostics should be written.
        parallel_assembly : bool, optional
            If True, the element-by-element assembly runs in parallel
            (multiprocessing Pool).  Defaults to False for determinism/easier
            debugging; turn on once everything is stable.

        Returns
        -------
        K_global : scipy.sparse.csr_matrix
        F_global : numpy.ndarray
            Flattened 1-D force vector.
        """
        logger.info("🔧 Assembling global stiffness and force matrices...")

        
        # 1. Work out the total number of DOFs for the model
        
        num_nodes = len(self.mesh_dictionary["node_ids"])
        total_dof  = num_nodes * 6        # -- assuming 6 DOFs per node

        
        # 2. Instantiate the high-performance assembler and run it
        
        assembler = AssembleGlobalSystem(
            elements                   = self.elements,
            element_stiffness_matrices = self.element_stiffness_matrices,
            element_force_vectors      = self.element_force_vectors,
            total_dof                  = total_dof,
            job_results_dir            = job_results_dir,
            parallel_assembly          = parallel_assembly            # <- switch on/off
        )

        try:
            K_global, F_global = assembler.assemble()
        except Exception as exc:                     # already logged internally
            logger.error("❌ Assembly failed – see assembly.log for details")
            raise                                      # re-raise so callers know

        
        # 3. Assemble global system diagnostics
        
        F_global = np.asarray(F_global).ravel()       # ensure 1-D

        log_system_diagnostics(
            K_global, F_global,
            bc_dofs         = [],                     
            job_results_dir = job_results_dir,
            label           = "Global System"
        )

        logger.info("✅ Global stiffness matrix and force vector successfully assembled")
        return K_global, F_global

    
    # -------------------------------------------------------------------------
    # 2) MODIFY GLOBAL MATRICES (BOUNDARY CONDITIONS)
    # -------------------------------------------------------------------------

    def modify_global_system(self, K_global, F_global, job_results_dir: str, *, fixed_dofs: np.ndarray | None = None):
        """
        Apply boundary conditions to the global system using the high-performance
        `ModifyGlobalSystem` helper.

        Parameters
        ----------
        K_global : scipy.sparse.csr_matrix
            Assembled global stiffness matrix.
        F_global : np.ndarray
            Assembled global force vector (1-D or flattenable to 1-D).
        job_results_dir : str
            Directory for boundary-condition logs/diagnostics.
        fixed_dofs : array-like[int], optional
            Explicit list of constrained DOF indices.  If omitted the helper
            defaults to the first six DOFs.

        Returns
        -------
        K_mod : scipy.sparse.csr_matrix
        F_mod : np.ndarray
        bc_dofs : np.ndarray
            The DOF indices actually constrained.
        """
        logger.info("🔒 Applying boundary conditions to global matrices…")

        
        # 1. Instantiate the boundary-condition helper
        
        modifier = ModifyGlobalSystem(
            K_global      = K_global,
            F_global      = np.asarray(F_global).ravel(),  # ensure 1-D
            job_results_dir = job_results_dir,
            fixed_dofs      = fixed_dofs                   # may be None
        )

        
        # 2. Run the boundary-condition pipeline
        
        try:
            K_mod, F_mod, bc_dofs = modifier.apply_boundary_conditions()
        except Exception as exc:                           # already logged inside
            logger.error("❌ Boundary-condition step failed – see boundary_conditions.log")
            raise                                           # propagate to caller

        
        # 3. Project-specific diagnostics (optional)
        
        log_system_diagnostics(
            K_mod, F_mod,
            bc_dofs         = bc_dofs,
            job_results_dir = job_results_dir,
            label           = "Modified System"
        )

        logger.info("✅ Boundary conditions successfully applied")
        return K_mod, F_mod, bc_dofs

    # -------------------------------------------------------------------------
    # 3) CONDENSE GLOBAL SYSTEM
    # -------------------------------------------------------------------------

    def condense_global_system(
            self,
            K_mod,
            F_mod,
            bc_dofs,
            job_results_dir: str,
            *,
            base_tol: float = 1e-12
        ):
        """
        Perform static condensation on the boundary-conditioned system using the
        high-performance `CondenseGlobalSystem` helper.

        Parameters
        ----------
        K_mod : scipy.sparse.csr_matrix
            Global stiffness matrix *after* boundary conditions have been applied.
        F_mod : np.ndarray
            Corresponding force vector (flattened to 1-D internally).
        bc_dofs : np.ndarray
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
        logger.info("📉 Condensing global system…")

        
        # 1.  Instantiate the condensation helper
        
        condenser = CondenseGlobalSystem(
            K_mod        = K_mod,
            F_mod        = np.asarray(F_mod).ravel(),   # ensure 1-D
            fixed_dofs   = bc_dofs,
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
        
        log_system_diagnostics(
            K_cond, F_cond,
            bc_dofs         = condensed_dofs,           # what’s left after BCs
            job_results_dir = job_results_dir,
            label           = "Condensed System"
        )

        logger.info(
            "✅ Static condensation complete "
            f"({len(condensed_dofs)} DOFs retained, "
            f"{len(inactive_dofs)} pruned)"
        )
        return condensed_dofs, inactive_dofs, K_cond, F_cond
    
    # -------------------------------------------------------------------------
    # 4) SOLVE CONDENSED SYSTEM
    # -------------------------------------------------------------------------

    def solve_condensed_system(self, K_cond, F_cond, 
                               job_results_dir: str, *,
                               solver_name: str = "cg",
                               preconditioner: str | None = None):
        """
        Solve ONLY the condensed linear system K_cond · U_cond = F_cond.
        Does NOT handle reconstruction to global DOFs.
    
        Parameters
        ----------
        K_cond, F_cond : csr_matrix, np.ndarray
            Condensed stiffness matrix / force vector
        job_results_dir : str
            Folder for solver logs and reports
        solver_name : str, optional
            Solver algorithm name (default "cg")
        preconditioner : str | None, optional
            Preconditioner type

        Returns
        -------
        U_cond : np.ndarray
            Displacements in condensed system ordering
        solver_report : dict
            Solver diagnostics report
        """
        logger.info("🧮 Solving condensed system…")

        # 1. Instantiate and run solver
        cond_solver = SolveCondensedSystem(
            K_cond=K_cond,
            F_cond=F_cond,
            solver_name=solver_name,
            job_results_dir=job_results_dir,
            preconditioner=preconditioner
        )
    
        U_cond = cond_solver.solve_condensed()
        if U_cond is None:
            raise RuntimeError("Condensed solver failed — check condensed_solver.log")

        # 2. Generate and save solver report
        solver_report = cond_solver.generate_condensed_report()
        report_path = Path(job_results_dir) / "condensed_solver_report.json"
        with open(report_path, "w", encoding="utf-8") as fp:
            json.dump(solver_report, fp, indent=2)
    
        logger.info(f"📑 Solver report saved to {report_path}")
        logger.info("✅ Condensed system successfully solved")
    
        return U_cond, solver_report

    # ---------------------------------------------------------------------------------
    # 5) RECONSTRUCT GLOBAL SYSTEM
    # ---------------------------------------------------------------------------------

    def reconstruct_global_system(self, condensed_dofs: np.ndarray, 
                                U_cond: np.ndarray, 
                                total_dof: int, 
                                job_results_dir: str, *,
                                fixed_dofs: np.ndarray | None = None,
                                inactive_dofs: np.ndarray | None = None):
        """
        Reconstruct full displacement vector and perform validation checks
    
        Parameters
        ----------
        condensed_dofs : np.ndarray
            Active DOF indices in original numbering
        U_cond : np.ndarray
            Solved displacements in condensed ordering
        total_dof : int
            Total DOFs in original system
        job_results_dir : str
            Directory for reconstruction logs
        fixed_dofs : np.ndarray | None, optional
            Fixed DOF indices for validation
        inactive_dofs : np.ndarray | None, optional
            Inactive DOFs eliminated during condensation

        Returns
        -------
        U_global : np.ndarray
            Full displacement vector in original ordering
        """
        logger.info("🔄 Reconstructing global displacement vector…")

        # 1. Instantiate reconstructor
        reconstructor = ReconstructGlobalSystem(
            active_dofs=condensed_dofs,
            U_cond=U_cond,
            total_dofs=total_dof,
            job_results_dir=Path(job_results_dir),
            fixed_dofs=fixed_dofs,
            inactive_dofs=inactive_dofs  # Pass for validation
        )

        # 2. Run reconstruction pipeline
        try:
            U_global = reconstructor.reconstruct()
        except Exception as exc:
            logger.error("❌ Reconstruction failed – see reconstruction.log")
            raise

        # 3. Run diagnostics on reconstructed solution
        reconstructor.validate_solution(U_global)
    
        # 4. Save reconstruction diagnostics
        recon_report = reconstructor.generate_report()
        report_path = Path(job_results_dir) / "reconstruction_report.json"
        with open(report_path, "w", encoding="utf-8") as fp:
            json.dump(recon_report, fp, indent=2)
    
        logger.info(
            f"✅ Displacement reconstruction complete "
            f"(min={U_global.min():.3e}, max={U_global.max():.3e})"
        )
        return U_global
    
    # -------------------------------------------------------------------------
    # 6) PRIMARY RESULTS PIPELINE
    # -------------------------------------------------------------------------
    def compute_primary_results(self, *, parallel_disassembly: bool = False):
        """
        Compute and save primary results:
        1. Global system matrices and solution
        2. Element-wise disassembled results
        """
        logger.info("📊 Computing primary results...")
        
        # 1. Compute global-level results (K, F, U, reactions)
        computer = ComputePrimaryResults(
            K_global=self.K_global,
            F_global=self.F_global,
            K_mod=self.K_mod,
            F_mod=self.F_mod,
            U_global=self.U_global,
            mesh_dict=self.mesh_dictionary
        )
        global_set = computer.compute()
        
        # 2. Disassemble to element level
        disassembler = DisassembleGlobalSystem(
            elements=self.elements,
            K_mod=self.K_mod,
            F_mod=self.F_mod,
            U_global=self.U_global,
            R_global=global_set.R_global,
            job_results_dir=self.primary_results_dir,
            parallel=parallel_disassembly
        )
        K_e_mod, F_e_mod, U_e, R_e = disassembler.disassemble()
        
        # 3. Package element results
        element_set = {
            "K_e": self.element_stiffness_matrices,
            "F_e": self.element_force_vectors,
            "K_e_mod": K_e_mod,
            "F_e_mod": F_e_mod,
            "U_e": U_e,
            "R_e": R_e,
        }
        
        # 4. Save all results
        SavePrimaryResults(
            job_name=self.job_name,
            start_timestamp=self.start_time,
            output_root=self.primary_results_dir,
            global_set=global_set,
            element_set=element_set
        ).write_all()
        
        # 5. Store in memory
        self.primary_results = {
            "global": global_set,
            "element": element_set
        }
        logger.info("✅ Primary results computed and saved")

    # -------------------------------------------------------------------------
    # 7) SECONDARY RESULTS PIPELINE
    # -------------------------------------------------------------------------
    def compute_secondary_results(self):
        """Compute and save secondary results (stresses, strains, etc.)"""
        if not self.primary_results:
            logger.error("❌ Compute primary results first!")
            raise RuntimeError("Primary results not available")
        
        logger.info("📈 Computing secondary results...")
        
        # 1. Compute secondary results
        computer = ComputeSecondaryResults(
            primary_results=self.primary_results["global"],
            mesh_dict=self.mesh_dictionary,
            elements=self.elements
        )
        secondary_set = computer.compute()
        
        # 2. Save results
        SaveSecondaryResults(
            job_name=self.job_name,
            start_timestamp=self.start_time,
            output_root=self.secondary_results_dir,
            secondary_set=secondary_set
        ).write_all()
        
        # 3. Store in memory
        self.secondary_results = secondary_set
        logger.info("✅ Secondary results computed and saved")

    # -------------------------------------------------------------------------
    # TOP-LEVEL SIMULATION FLOW
    # -------------------------------------------------------------------------
    def run(self, *, parallel_assembly=False, parallel_disassembly=False):
        """Execute full static simulation workflow"""
        try:
            self.setup_simulation()

            # 0. Prepare local system
            self.prepare_local_system(job_results_dir=self.primary_results_dir)

            
            # 1. Assemble global system
            self.K_global, self.F_global = self.assemble_global_system(
                self.primary_results_dir,
                parallel_assembly=parallel_assembly
            )
            
            # 2. Apply boundary conditions
            self.K_mod, self.F_mod, self.bc_dofs = self.modify_global_system(
                self.K_global,
                self.F_global,
                self.primary_results_dir
            )
            
            # 3. Condense system
            (self.condensed_dofs, 
             self.inactive_dofs, 
             self.K_cond, 
             self.F_cond) = self.condense_global_system(
                self.K_mod,
                self.F_mod,
                self.bc_dofs,
                self.primary_results_dir
            )
            
            # 4. Solve condensed system
            self.U_cond, _ = self.solve_condensed_system(
                self.K_cond,
                self.F_cond,
                self.primary_results_dir
            )
            
            # 5. Reconstruct global solution
            total_dof = len(self.mesh_dictionary["node_ids"]) * 6
            self.U_global = self.reconstruct_global_system(
                self.condensed_dofs,
                self.U_cond,
                total_dof,
                self.primary_results_dir,
                fixed_dofs=self.bc_dofs,
                inactive_dofs=self.inactive_dofs
            )
            
            # 6. Compute primary results
            self.compute_primary_results(
                parallel_disassembly=parallel_disassembly
            )
            
            # 7. Compute secondary results
            self.compute_secondary_results()
            
            logger.info(f"🏁 Simulation completed successfully in {self.results_root}")
            
        except Exception as e:
            logger.exception("💥 Simulation failed with critical error")
            raise RuntimeError("Simulation aborted") from e
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from typing import List, Tuple, Optional
import logging
import os
import time
from functools import partial
from pathlib import Path

from processing_OOP.static.results.containers.map_results import MapEntry

FLOAT_FORMAT = "%.17e"

class AssembleGlobalSystem:
    """
    Assembles the global stiffness matrix and force vector for a finite element system.

    Supports input validation, parallel assembly, symmetry checks, logging, and saving.
    """

    def __init__(
        self,
        elements: List[object],
        element_stiffness_matrices: Optional[List[coo_matrix]] = None,
        element_force_vectors: Optional[List[np.ndarray]] = None,
        total_dof: Optional[int] = None,
        job_results_dir: Optional[str] = None,
        symmetry_tol: float = 1e-8
    ):
        """
        Parameters
        ----------
        elements : list of object
            List of finite element objects with an `assemble_global_dof_indices()` method.
        element_stiffness_matrices : list of coo_matrix, optional
            Local element stiffness matrices.
        element_force_vectors : list of ndarray, optional
            Local element force vectors.
        total_dof : int, optional
            Total number of global degrees of freedom.
        job_results_dir : str, optional
            Path to output logging directory.
        symmetry_tol : float, default=1e-8
            Tolerance for symmetry check on global matrix.
        """

        self.float_format = FLOAT_FORMAT

        self.elements = elements
        self.element_stiffness_matrices = element_stiffness_matrices
        self.element_force_vectors = element_force_vectors
        self.total_dof = total_dof
        self.job_results_dir = Path(job_results_dir) if job_results_dir else None
        self.symmetry_tol = symmetry_tol
        self.K_global: Optional[csr_matrix] = None
        self.F_global: Optional[np.ndarray] = None
        self.local_global_dof_map = None
        self.assembly_time = None
        self.logger = self._init_logging()

    def _init_logging(self):
        """Initialize logger with optional file output."""
        logger = logging.getLogger(f"AssembleGlobalSystem.{id(self)}")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        log_path = None
        if self.job_results_dir:
            logs_dir = self.job_results_dir.parent / "logs"  # ✅ Place logs next to primary_results
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / "AssembleGlobalSystem.log"

            try:
                file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
                file_handler.setFormatter(logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s "
                    "(Module: %(module)s, Line: %(lineno)d)"
                ))
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"⚠️ Failed to create file handler for AssembleGlobalSystem class log: {e}")

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(stream_handler)

        if log_path:
            logger.debug(f"📁 Log file created at: {log_path}")

        return logger

    def assemble(self) -> Tuple[csr_matrix, np.ndarray, list[np.ndarray]]:
        """
        Assemble the global stiffness matrix and force vector.

        Returns
        -------
        tuple
            (K_global, F_global, local_global_dof_map) where  
            • **K_global** – csr_matrix, global stiffness  
            • **F_global** – (N,) float64 array, global load vector  
            • **local_global_dof_map** – list of 1-D int32 arrays, one per element,
              mapping each local DOF index (0…nᵢ-1) to its global DOF.
        """
        start_time = time.perf_counter()
        self.logger.info("🔧 Starting global matrix assembly...")

        try:
            self._validate_inputs()
            self._compute_local_global_dof_map()
            self._build_assembly_map()
            self._log_system_info()
            self._assemble_stiffness_matrix()
            self._assemble_force_vector()
            self._log_matrix_contents()
            self._export_K_global()
            self._export_F_global()
        except Exception as e:
            self.logger.critical(f"❌ Assembly failed: {str(e)}", exc_info=True)
            raise RuntimeError("Global assembly failed") from e

        self.assembly_time = time.perf_counter() - start_time
        self._log_performance_metrics()
        self.logger.info("✅ Assembly complete.")

        self.assembly_map = self._build_assembly_map()

        return self.K_global, self.F_global, self.local_global_dof_map, self.assembly_map

    def _validate_inputs(self):
        """Validate elements, stiffness matrices, and force vectors."""
        if not self.elements:
            raise ValueError("No elements provided")
        if self.total_dof is None or self.total_dof <= 0:
            raise ValueError("Invalid total_dof - must be positive integer")
        if self.element_stiffness_matrices and len(self.element_stiffness_matrices) != len(self.elements):
            raise ValueError("Stiffness matrices count doesn't match elements")
        if self.element_force_vectors and len(self.element_force_vectors) != len(self.elements):
            raise ValueError("Force vectors count doesn't match elements")
        if self.element_stiffness_matrices:
            for i, Ke in enumerate(self.element_stiffness_matrices):
                if Ke.nnz > 0 and not np.isfinite(Ke.data).all():
                    raise ValueError(f"Non-finite values in stiffness matrix {i}")
        if self.element_force_vectors:
            for i, Fe in enumerate(self.element_force_vectors):
                if not np.isfinite(Fe).all():
                    raise ValueError(f"Non-finite values in force vector {i}")

    def _compute_local_global_dof_map(self):
        """Compute global DOF mappings for each element."""
        self.local_global_dof_map = []
        validation_errors = []
        mapping_records = []

        for elem in self.elements:
            try:
                element_id = int(elem.element_id)  # Ensure Python int
                dof = elem.assemble_global_dof_indices(element_id)
                validated_dof = np.asarray(dof, dtype=np.int32).ravel()  # contiguous int32 buffer


                if validated_dof.size == 0:
                    raise ValueError("Empty DOF mapping")
                if validated_dof.min() < 0:
                    raise ValueError(f"Negative DOF index: {validated_dof.min()}")
                if validated_dof.max() >= self.total_dof:
                    raise ValueError(f"DOF index {validated_dof.max()} >= total_dof {self.total_dof}")
                if len(np.unique(validated_dof)) != len(validated_dof):
                    raise ValueError("Duplicate DOF indices detected")

                # ✅ Store mapping for global assembly
                self.local_global_dof_map.append(validated_dof)

                # 📋 Record for output/logging
                mapping_records.append({
                    "Element ID": element_id,
                    "Local DOF": list(range(len(validated_dof))),  # usually [0,1,…,11]
                    "Global DOF": validated_dof.tolist()
                })

                self.logger.debug(f"Element {element_id} DOF mapping validated")

            except Exception as e:
                elem_id = getattr(elem, 'element_id', f"Element_{id(elem)}")
                validation_errors.append({
                    'element_id': int(elem_id) if isinstance(elem_id, (np.integer, int)) else str(elem_id),
                    'error_type': type(e).__name__,
                    'message': str(e),
                    'dof_indices': dof if 'dof' in locals() else None
                })
                self.logger.error(f"Element {elem_id} failed validation: {type(e).__name__} - {e}")

        if validation_errors:
            msgs = [f"{e['element_id']}: {e['error_type']} - {e['message']}" for e in validation_errors[:10]]
            if len(validation_errors) > 10:
                msgs.append(f"... {len(validation_errors) - 10} additional errors")
            raise RuntimeError("Assembly aborted due to invalid DOF mappings:\n" + "\n".join(msgs))

        # Summarise all mappings
        df_dof = pd.DataFrame(mapping_records)
        self.logger.debug("\n✅ DOF Mappings Summary:\n" + df_dof.to_string(index=False))

        # Save DOF mapping CSV to /maps directory
        maps_path = None
        if self.job_results_dir:
            maps_dir = self.job_results_dir.parent / "maps"
            maps_dir.mkdir(parents=True, exist_ok=True)
            maps_path = maps_dir / "01_assembly_map.csv"
            #try:
                #df_dof.to_csv(maps_path, index=False)
                #self.logger.info(f"🗂️ DOF mapping saved to: {maps_path}")
            #except Exception as e:
                #self.logger.warning(f"⚠️ Failed to save DOF mapping CSV: {e}")

        self.mapping_records = mapping_records

    def _coo_to_dataframe(self, matrix: coo_matrix, value_label: str = "Value") -> pd.DataFrame:
        """Return a DataFrame from a COO matrix with human-readable labels."""
        return pd.DataFrame({
            "Row": matrix.row.tolist(),
            "Col": matrix.col.tolist(),
            value_label: matrix.data
        })

    def _assemble_stiffness_matrix(self):
        """Assemble the global stiffness matrix in CSR format."""
        if not self.element_stiffness_matrices:
            self.K_global = csr_matrix((self.total_dof, self.total_dof), dtype=np.float64)
            return

        assembly_func = partial(self._process_stiffness_element)
        elements = list(zip(self.element_stiffness_matrices, self.local_global_dof_map))

        results = [assembly_func(Ke, dof) for Ke, dof in elements]

        all_rows = np.concatenate([r for r, _, _ in results]).astype(np.int32)
        all_cols = np.concatenate([c for _, c, _ in results]).astype(np.int32)
        all_data = np.concatenate([d for _, _, d in results])

        self.K_global = coo_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(self.total_dof, self.total_dof),
            dtype=np.float64
        ).tocsr()
        self.logger.info("✅ Stiffness matrix assembled")

    def _assemble_force_vector(self):
        """
        Assemble the global force vector with float64 precision.

        Raises
        ------
        ValueError
            If an element's force vector size does not match its DOF mapping.

        Postconditions
        --------------
        self.F_global : np.ndarray
            Global force vector with shape (total_dof,) and dtype float64.
        """
        if not self.element_force_vectors:
            self.F_global = np.zeros(self.total_dof, dtype=np.float64)
        else:
            F = np.zeros(self.total_dof, dtype=np.float64)
            for idx, (Fe, dof_map) in enumerate(zip(self.element_force_vectors, self.local_global_dof_map)):
                if Fe.shape[0] != dof_map.size:
                    raise ValueError(f"Force vector {idx} shape mismatch")
                F[dof_map] += Fe.astype(np.float64)
            self.F_global = F

        assert isinstance(self.F_global, np.ndarray)
        assert self.F_global.shape == (self.total_dof,)
        assert self.F_global.dtype == np.float64
        self.logger.info("✅ Force vector assembled")

    def _process_stiffness_element(self, Ke: coo_matrix, dof_map: np.ndarray):
        """Return global row, col, and data arrays for one element matrix."""
        if dof_map.size == 0:
            return np.array([]), np.array([]), np.array([])

        # Ensure matrix is in COO format
        if not isinstance(Ke, coo_matrix):
            Ke = Ke.tocoo()

        if Ke.shape != (dof_map.size, dof_map.size):
            raise ValueError(f"Matrix shape {Ke.shape} does not match DOF size {dof_map.size}")

        rows = dof_map[Ke.row]
        cols = dof_map[Ke.col]
        return rows.astype(np.int32), cols.astype(np.int32), Ke.data.astype(np.float64)

    def _is_symmetric(self, tol: float = 1e-8) -> bool:
        """
        Check if the global stiffness matrix is symmetric.

        Parameters
        ----------
        tol : float
            Tolerance for symmetry check.

        Returns
        -------
        bool
            True if matrix is symmetric within tolerance, else False.
        """
        if self.K_global is None:
            return False
        diff = self.K_global - self.K_global.T
        return np.abs(diff.data).max() < tol if diff.nnz > 0 else True

    def _export_K_global(self):
        """Export K_global to <primary_results>/K_global.csv with full precision."""
        if not self.job_results_dir or self.K_global is None:
            return

        K_path = self.job_results_dir / "01_K_global.csv"
        K_coo  = self.K_global.tocoo()

        
        #df_K = self._coo_to_dataframe(K_coo, value_label="K Value")
        #df_K.to_csv(K_path, index=False, float_format=self.float_format)
        #self.logger.info(f"💾 Global stiffness matrix saved to: {K_path}")

    def _export_F_global(self):
        """Export F_global to <primary_results>/F_global.csv with full precision."""
        if not self.job_results_dir or self.F_global is None:
            return

        F_path = self.job_results_dir / "02_F_global.csv"

        # Again, Python ints for DOF indices
        #df_F = pd.DataFrame({
            #"DOF":   list(range(self.total_dof)),
            #"F Value": self.F_global           # float64
        #})
        #df_F.to_csv(F_path, index=False, float_format=self.float_format)
        #self.logger.info(f"💾 Global force vector saved to: {F_path}")

    def _log_system_info(self):
        """Log basic information about the system before assembly."""
        self.logger.info(
            f"🧾 System Info:\n"
            f"  Total DOFs: {self.total_dof}\n"
            f"  Number of elements: {len(self.elements)}\n"
            f"  Symmetry tolerance: {self.symmetry_tol:.1e}"
        )

    def _log_performance_metrics(self):
        """Log overall timing and performance of the assembly."""
        stats = [
            f"Assembly Time: {self.assembly_time:.2f}s",
            f"Elements Processed: {len(self.elements)}",
            f"Matrix Density: {self.K_global.nnz / (self.total_dof**2):.2e}",
            f"Memory Usage: {self.K_global.data.nbytes / 1e6:.2f}MB",
            f"Processing Rate: {len(self.elements)/self.assembly_time:.1f} elements/s"
        ]
        self.logger.info("📊 Performance Metrics:\n  " + "\n  ".join(stats))

    def _log_matrix_contents(self):
        """Log global matrix summary and detailed contents if appropriate."""
        self.logger.info(
            f"Global Stiffness Matrix:\n"
            f"  Non-zero entries: {self.K_global.nnz}\n"
            f"  Symmetry: {self._is_symmetric()}\n"
            f"  Norm: {np.linalg.norm(self.K_global.data):.3e}"
        )

        coo = self.K_global.tocoo()
        n_dof = self.K_global.shape[0]

        if n_dof <= 100:
            # Log full matrix as dense DataFrame
            df_K = pd.DataFrame(coo.toarray(), dtype=np.float64)
            df_K.index.name = "Row"
            df_K.columns.name = "Col"
            self.logger.debug("\nFull Stiffness Matrix (Dense):\n" + df_K.to_string(float_format="%.2e"))
        else:
            # Log sparse sample
            sample_size = min(1000, coo.nnz)
            sample_df = self._coo_to_dataframe(coo, value_label="K Value").sample(n=sample_size, random_state=0)
            self.logger.debug("\nStiffness Matrix Sample (COO):\n" + sample_df.to_string(index=False))
    
    def _build_assembly_map(self) -> list[MapEntry]:
        """
        Construct initial map from element local DOFs to global DOFs.
        All DOFs assumed free and active at this stage.
        """
        return [
            MapEntry(
                element_id=i,
                local_dof=np.arange(len(global_dofs), dtype=np.int32),
                global_dof=global_dofs,
                fixed_flag=np.zeros(len(global_dofs), dtype=np.int32),
                zero_flag=np.zeros(len(global_dofs), dtype=np.int32),
                active_flag=np.ones(len(global_dofs), dtype=np.int32),
                condensed_dof=np.full(len(global_dofs), "", dtype=object)
            )
            for i, global_dofs in enumerate(self.local_global_dof_map)
        ]
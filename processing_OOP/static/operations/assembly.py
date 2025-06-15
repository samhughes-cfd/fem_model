import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from typing import List, Tuple, Optional, Dict, Callable
import logging
import os
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path


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
        parallel_assembly: bool = False,
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
        parallel_assembly : bool, default=False
            Use multiprocessing for stiffness matrix assembly.
        symmetry_tol : float, default=1e-8
            Tolerance for symmetry check on global matrix.
        """
        self.elements = elements
        self.element_stiffness_matrices = element_stiffness_matrices
        self.element_force_vectors = element_force_vectors
        self.total_dof = total_dof
        self.job_results_dir = Path(job_results_dir) if job_results_dir else None
        self.parallel_assembly = parallel_assembly
        self.symmetry_tol = symmetry_tol
        self.K_global: Optional[csr_matrix] = None
        self.F_global: Optional[np.ndarray] = None
        self.dof_mappings = None
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

    def assemble(self) -> Tuple[csr_matrix, np.ndarray]:
        """
        Assemble the global stiffness matrix and force vector.

        Returns
        -------
        K_global : csr_matrix
            Assembled global stiffness matrix.
        F_global : ndarray
            Assembled global force vector.
        """
        start_time = time.perf_counter()
        self.logger.info("🔧 Starting global matrix assembly...")

        try:
            self._validate_inputs()
            self._compute_dof_mappings()
            self._log_system_info()
            self._assemble_stiffness_matrix()
            self._assemble_force_vector()
            self._post_process()
            self._log_matrix_contents()
        except Exception as e:
            self.logger.critical(f"❌ Assembly failed: {str(e)}", exc_info=True)
            raise RuntimeError("Global assembly failed") from e

        self.assembly_time = time.perf_counter() - start_time
        self._log_performance_metrics()
        self.logger.info("✅ Assembly complete.")
        return self.K_global, self.F_global

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

    def _compute_dof_mappings(self):
        """Compute global DOF mappings for each element."""
        self.dof_mappings = []
        validation_errors = []
        mapping_records = []

        for elem in self.elements:
            try:
                element_id = int(elem.element_id)  # Ensure Python int
                dof = elem.assemble_global_dof_indices(element_id)
                validated_dof = np.asarray(dof, dtype=np.int32).flatten()

                if validated_dof.size == 0:
                    raise ValueError("Empty DOF mapping")
                if validated_dof.min() < 0:
                    raise ValueError(f"Negative DOF index: {validated_dof.min()}")
                if validated_dof.max() >= self.total_dof:
                    raise ValueError(f"DOF index {validated_dof.max()} >= total_dof {self.total_dof}")
                if len(np.unique(validated_dof)) != len(validated_dof):
                    raise ValueError("Duplicate DOF indices detected")

                self.dof_mappings.append(validated_dof)
                mapping_records.append({
                    "Element ID": element_id,
                    "DOF Indices": validated_dof.tolist()
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

        # Log all validated DOF mappings as a DataFrame
        df_dof = pd.DataFrame(mapping_records)
        self.logger.debug("\n✅ DOF Mappings Summary:\n" + df_dof.to_string(index=False))

        if self.job_results_dir:
            mapping_csv_path = self.job_results_dir / "assembly_dof_mapping.csv"
            try:
                df_dof.to_csv(mapping_csv_path, index=False)
                self.logger.info(f"🗂️ DOF mapping saved to: {mapping_csv_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to save DOF mapping CSV: {e}")

    def _assemble_stiffness_matrix(self):
        """Assemble the global stiffness matrix in CSR format."""
        if not self.element_stiffness_matrices:
            self.K_global = csr_matrix((self.total_dof, self.total_dof), dtype=np.float64)
            return

        assembly_func = partial(self._process_stiffness_element)
        elements = list(zip(self.element_stiffness_matrices, self.dof_mappings))

        try:
            if self.parallel_assembly:
                # Disallow parallelism inside daemonic processes
                if multiprocessing.current_process().daemon:
                    self.logger.warning("⚠️ Detected daemon context: disabling parallel assembly.")
                    raise RuntimeError("Daemon context detected")

                with Pool(cpu_count()) as pool:
                    results = pool.starmap(assembly_func, elements)
            else:
                results = [assembly_func(Ke, dof) for Ke, dof in elements]

        except (RuntimeError, AssertionError) as e:
            self.logger.warning(f"⚠️ Parallel assembly failed: {e}. Falling back to serial execution.")
            results = [assembly_func(Ke, dof) for Ke, dof in elements]

        all_rows = np.concatenate([r for r, _, _ in results])
        all_cols = np.concatenate([c for _, c, _ in results])
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
            for idx, (Fe, dof_map) in enumerate(zip(self.element_force_vectors, self.dof_mappings)):
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

    def _post_process(self):
        """Remove duplicates and zeros from K_global."""
        self.K_global.sum_duplicates()
        self.K_global.eliminate_zeros()

    def _log_system_info(self):
        """Log basic information about the system before assembly."""
        self.logger.info(
            f"🧾 System Info:\n"
            f"  Total DOFs: {self.total_dof}\n"
            f"  Number of elements: {len(self.elements)}\n"
            f"  Parallel assembly: {self.parallel_assembly}\n"
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
        """Log global matrix summary."""
        self.logger.info(
            f"Global Stiffness Matrix:\n"
            f"  Non-zero entries: {self.K_global.nnz}\n"
            f"  Symmetry: {self._is_symmetric()}\n"
            f"  Norm: {np.linalg.norm(self.K_global.data):.3e}"
        )

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
            sample_df = pd.DataFrame({
                "Row": coo.row,
                "Col": coo.col,
                "Value": coo.data
            }).sample(n=sample_size, random_state=0)
            self.logger.debug("\nStiffness Matrix Sample (COO):\n" + sample_df.to_string(index=False))
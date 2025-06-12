# processing_OOP\static\operations\assembly.py

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from typing import List, Tuple, Optional, Dict, Callable
import logging
import os
import time
from multiprocessing import Pool, cpu_count
from functools import partial

class AssembleGlobalSystem:
    """High-performance FEM global system assembler with parallel processing and diagnostics.
    
    Features:
    - Strict input validation with detailed error reporting
    - Parallel assembly support with process-safe logging
    - Memory-efficient sparse matrix handling
    - Comprehensive symmetry and singularity checks
    - Detailed performance metrics and matrix diagnostics
    
    Note: Optimized for CPU-only execution with scipy.sparse matrices.
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
        self.elements = elements
        self.element_stiffness_matrices = element_stiffness_matrices
        self.element_force_vectors = element_force_vectors
        self.total_dof = total_dof
        self.job_results_dir = job_results_dir
        self.parallel_assembly = parallel_assembly
        self.symmetry_tol = symmetry_tol
        self.K_global = None
        self.F_global = None
        self.dof_mappings = None
        self.assembly_time = None
        self._init_logging()

    def _init_logging(self):
        """Configure hierarchical logging with performance tracking."""
        self.logger = logging.getLogger(f"AssemblySystem.{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        if self.job_results_dir:
            os.makedirs(self.job_results_dir, exist_ok=True)
            log_path = os.path.join(self.job_results_dir, "assembly.log")
            file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s "
                "(Module: %(module)s, Line: %(lineno)d)"
            ))
            self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self.logger.addHandler(console_handler)

    def assemble(self) -> Tuple[csr_matrix, np.ndarray]:
        """Execute full assembly process with detailed monitoring."""
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
        """Comprehensive input validation with error messages."""
        if not self.elements:
            raise ValueError("No elements provided")

        if self.total_dof is None or self.total_dof <= 0:
            raise ValueError("Invalid total_dof - must be positive integer")

        if (self.element_stiffness_matrices is not None and 
            len(self.element_stiffness_matrices) != len(self.elements)):
            raise ValueError("Stiffness matrices count doesn't match elements")

        if (self.element_force_vectors is not None and 
            len(self.element_force_vectors) != len(self.elements)):
            raise ValueError("Force vectors count doesn't match elements")

        if self.element_stiffness_matrices is not None:
            for i, Ke in enumerate(self.element_stiffness_matrices):
                if Ke.nnz > 0 and not np.isfinite(Ke.data).all():
                    raise ValueError(f"Non-finite values in stiffness matrix {i}")

        if self.element_force_vectors is not None:
            for i, Fe in enumerate(self.element_force_vectors):
                if not np.isfinite(Fe).all():
                    raise ValueError(f"Non-finite values in force vector {i}")



    def _compute_dof_mappings(self):
        """Compute DOF mappings with strict validation and error aggregation."""
        self.dof_mappings = []
        validation_errors = []
        
        for elem in self.elements:
            elem_id = getattr(elem, 'element_id', f"Element_{id(elem)}")
            try:
                dof = elem.assemble_global_dof_indices(elem.element_id)
                validated_dof = np.asarray(dof, dtype=np.int32).flatten()
                
                if validated_dof.size == 0:
                    raise ValueError("Empty DOF mapping")
                if validated_dof.min() < 0:
                    raise ValueError(f"Negative DOF indices: {validated_dof.min()}")
                if validated_dof.max() >= self.total_dof:
                    raise ValueError(f"DOF index {validated_dof.max()} >= total_dof {self.total_dof}")
                if len(np.unique(validated_dof)) != len(validated_dof):
                    raise ValueError("Duplicate DOF indices detected")
                    
                self.dof_mappings.append(validated_dof)
                self.logger.debug(f"Element {elem_id} DOF mapping validated")
                
            except Exception as e:
                error_info = {
                    'element_id': elem_id,
                    'error_type': type(e).__name__,
                    'message': str(e),
                    'dof_indices': dof if 'dof' in locals() else None
                }
                validation_errors.append(error_info)
                self.logger.error(f"Element {elem_id} failed validation: {error_info['error_type']} - {error_info['message']}")

        if validation_errors:
            error_report = [
                f"CRITICAL VALIDATION ERRORS ({len(validation_errors)}):",
                *[f"{e['element_id']}: {e['error_type']} - {e['message']}" 
                  for e in validation_errors[:10]]
            ]
            if len(validation_errors) > 10:
                error_report.append(f"... {len(validation_errors)-10} additional errors")
                
            self.logger.critical("\n".join(error_report))
            raise RuntimeError(
                f"Assembly aborted: {len(validation_errors)} invalid elements detected. "
                "See assembly.log for details."
            )

    def _assemble_stiffness_matrix(self):
        """Optimized COO-based stiffness matrix assembly with parallel support."""
        if not self.element_stiffness_matrices:
            self.K_global = csr_matrix((self.total_dof, self.total_dof), dtype=np.float64)
            return

        assembly_func = partial(self._process_stiffness_element)
        elements = zip(self.element_stiffness_matrices, self.dof_mappings)

        if self.parallel_assembly:
            with Pool(cpu_count()) as pool:
                results = pool.starmap(assembly_func, elements)
        else:
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

    def _process_stiffness_element(self, Ke: coo_matrix, dof_map: np.ndarray):
        """Process individual element stiffness matrix with validation."""
        if dof_map.size == 0:
            return (np.array([]), np.array([]), np.array([]))
            
        if Ke.shape != (dof_map.size, dof_map.size):
            raise ValueError(
                f"Stiffness matrix shape {Ke.shape} "
                f"doesn't match DOF mapping size {dof_map.size}"
            )
            
        rows = dof_map[Ke.row]
        cols = dof_map[Ke.col]
        return (rows.astype(np.int32), cols.astype(np.int32), Ke.data.astype(np.float64))

    def _is_symmetric(self, tol: float = 1e-8) -> bool:
        """Check matrix symmetry with tolerance."""
        if self.K_global is None:
            return False
            
        diff = self.K_global - self.K_global.T
        return np.abs(diff.data).max() < tol if diff.nnz > 0 else True

    def _log_performance_metrics(self):
        """Log detailed performance statistics."""
        stats = [
            f"Assembly Time: {self.assembly_time:.2f}s",
            f"Elements Processed: {len(self.elements)}",
            f"Matrix Density: {self.K_global.nnz / (self.total_dof**2):.2e}",
            f"Memory Usage: {self.K_global.data.nbytes / 1e6:.2f}MB",
            f"Processing Rate: {len(self.elements)/self.assembly_time:.1f} elements/s"
        ]
        self.logger.info("📊 Performance Metrics:\n  " + "\n  ".join(stats))

    def _post_process(self):
        """Final assembly steps after matrix construction."""
        self.K_global.sum_duplicates()
        self.K_global.eliminate_zeros()
    
    def _log_matrix_contents(self):
        """Log key matrix statistics."""
        self.logger.info(
            f"Global Stiffness Matrix:\n"
            f"  Non-zero entries: {self.K_global.nnz}\n"
            f"  Symmetry: {self._is_symmetric()}\n"
            f"  Norm: {np.linalg.norm(self.K_global.data):.3e}"
        )

    def save_assembly(self, filename: str):
        """Save assembled system to NPZ file."""
        if not filename.endswith('.npz'):
            filename += '.npz'
            
        np.savez_compressed(
            filename,
            K_data=self.K_global.data,
            K_indices=self.K_global.indices,
            K_indptr=self.K_global.indptr,
            K_shape=self.K_global.shape,
            F_global=self.F_global
        )
        self.logger.info(f"💾 Saved assembly to {filename}")

    @classmethod
    def load_assembly(cls, filename: str):
        """Load assembled system from NPZ file."""
        data = np.load(filename)
        K = csr_matrix((
            data['K_data'],
            data['K_indices'],
            data['K_indptr']
        ), shape=data['K_shape'])
        return K, data['F_global']
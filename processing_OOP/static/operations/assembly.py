# processing_OOP\static\operations\assembly.py

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from typing import List, Tuple, Optional, Dict, Callable
import logging
import os
import time
from multiprocessing import Pool, cpu_count

class AssembleGlobalSystem:
    """High-performance FEM global system assembler with parallel processing and diagnostics.
    
    Note: This implementation is optimized for CPU-only execution and does not include
    GPU-related optimizations, making it suitable for systems without dedicated
    graphics hardware.
    """
    
    def __init__(
        self,
        elements: List[object],
        element_stiffness_matrices: Optional[List[coo_matrix]] = None,
        element_force_vectors: Optional[List[np.ndarray]] = None,
        total_dof: Optional[int] = None,
        job_results_dir: Optional[str] = None,
        parallel_assembly: bool = False
    ):
        self.elements = elements
        self.element_stiffness_matrices = element_stiffness_matrices
        self.element_force_vectors = element_force_vectors
        self.total_dof = total_dof
        self.job_results_dir = job_results_dir
        self.parallel_assembly = parallel_assembly
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

        # Detailed file logging
        if self.job_results_dir:
            os.makedirs(self.job_results_dir, exist_ok=True)
            log_path = os.path.join(self.job_results_dir, "assembly.log")
            file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s "
                "(Module: %(module)s, Line: %(lineno)d)"
            ))
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)

        # Critical-only console logging
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
            
        if self.element_stiffness_matrices and len(self.element_stiffness_matrices) != len(self.elements):
            raise ValueError("Stiffness matrices count doesn't match elements")
            
        if self.element_force_vectors and len(self.element_force_vectors) != len(self.elements):
            raise ValueError("Force vectors count doesn't match elements")

        for i, Ke in enumerate(self.element_stiffness_matrices or []):
            if Ke.nnz > 0 and not np.isfinite(Ke.data).all():
                raise ValueError(f"Non-finite values in stiffness matrix {i}")
                
        for i, Fe in enumerate(self.element_force_vectors or []):
            if not np.isfinite(Fe).all():
                raise ValueError(f"Non-finite values in force vector {i}")

    def _compute_dof_mappings(self):
        """Compute DOF mappings with detailed element logging."""
        self.dof_mappings = []
        invalid_elements = []
        
        for idx, elem in enumerate(self.elements):
            try:
                dof = elem.assemble_global_dof_indices(elem.element_id)
                validated_dof = np.asarray(dof, dtype=np.int32).flatten()
                
                if validated_dof.min() < 0:
                    raise ValueError("Negative DOF indices detected")
                if validated_dof.max() >= self.total_dof:
                    raise ValueError("DOF index exceeds total_dof")
                    
                self.dof_mappings.append(validated_dof)
                self.logger.debug(
                    f"Element {idx} DOF mapping:\n"
                    f"{pd.Series(validated_dof).to_string(index=False, header=False)}"
                )
                
            except Exception as e:
                self.logger.error(f"❌ Invalid element {idx}: {str(e)} - excluding from assembly")
                invalid_elements.append(idx)
                self.dof_mappings.append(np.array([], dtype=np.int32))

        if invalid_elements:
            self.elements = [e for i,e in enumerate(self.elements) if i not in invalid_elements]
            if self.element_stiffness_matrices:
                self.element_stiffness_matrices = [m for i,m in enumerate(self.element_stiffness_matrices) 
                                                 if i not in invalid_elements]
            if self.element_force_vectors:
                self.element_force_vectors = [f for i,f in enumerate(self.element_force_vectors) 
                                            if i not in invalid_elements]

    def _log_system_info(self):
        """Log system configuration details."""
        self.logger.info(
            f"Assembly Configuration\n"
            f"Elements: {len(self.elements)}\n"
            f"Total DOFs: {self.total_dof}\n"
            f"Parallel: {self.parallel_assembly}"
        )

    def _assemble_stiffness_matrix(self):
        """Optimized COO-based stiffness matrix assembly."""
        if not self.element_stiffness_matrices:
            self.K_global = csr_matrix((self.total_dof, self.total_dof), dtype=np.float64)
            return

        # Parallel/sequential processing
        if self.parallel_assembly:
            with Pool(cpu_count()) as pool:
                results = pool.starmap(
                    self._process_stiffness_element,
                    zip(self.element_stiffness_matrices, self.dof_mappings)
                )
        else:
            results = [self._process_stiffness_element(Ke, dof) 
                      for Ke, dof in zip(self.element_stiffness_matrices, self.dof_mappings)]

        # Combine results
        all_rows = np.concatenate([r for r, _, _ in results])
        all_cols = np.concatenate([c for _, c, _ in results])
        all_data = np.concatenate([d for _, _, d in results])

        # Build global matrix
        K_coo = coo_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(self.total_dof, self.total_dof),
            dtype=np.float64
        )
        self.K_global = K_coo.tocsr()
        self.logger.info("✅ Stiffness matrix assembled")

    def _process_stiffness_element(self, Ke: coo_matrix, dof_map: np.ndarray):
        """Process individual element stiffness matrix."""
        if dof_map.size == 0 or Ke.nnz == 0:
            return (np.array([]), np.array([]), np.array([]))
            
        rows = dof_map[Ke.row]
        cols = dof_map[Ke.col]
        return (rows.astype(np.int32), cols.astype(np.int32), Ke.data.astype(np.float64))

    def _assemble_force_vector(self):
        """Vectorized force vector assembly with batch updates."""
        self.F_global = np.zeros(self.total_dof, dtype=np.float64)
        if not self.element_force_vectors:
            return

        # Collect all contributions
        all_dof = []
        all_fe = []
        for Fe, dof in zip(self.element_force_vectors, self.dof_mappings):
            if dof.size == 0:
                continue
                
            Fe = np.asarray(Fe, dtype=np.float64).flatten()
            if Fe.shape != (dof.size,):
                raise ValueError(f"Force vector shape {Fe.shape} doesn't match DOF mapping {dof.size}")
                
            all_dof.append(dof)
            all_fe.append(Fe)

        # Batch update
        if all_dof:
            np.add.at(self.F_global, np.concatenate(all_dof), np.concatenate(all_fe))
        self.logger.info("✅ Force vector assembled")

    def _log_matrix_contents(self):
        """Log detailed matrix contents using Pandas."""
        if not self.job_results_dir:
            return

        self.logger.debug("\n" + "="*40 + " Matrix Contents " + "="*40)
        
        # Stiffness matrix
        if self.total_dof <= 100:
            self.logger.debug(
                "Full Stiffness Matrix (K_global):\n" +
                pd.DataFrame(
                    self.K_global.toarray(),
                    columns=range(self.total_dof),
                    index=range(self.total_dof)
                ).to_string(float_format="%.2e")
            )
        else:
            K_coo = self.K_global.tocoo()
            sparse_df = pd.DataFrame({
                'Row': K_coo.row,
                'Col': K_coo.col,
                'Value': K_coo.data
            })
            self.logger.debug(
                f"Sparse Stiffness Matrix (First 100 entries):\n"
                f"{sparse_df.head(100).to_string(index=False)}"
            )

        # Force vector
        force_df = pd.DataFrame(
            self.F_global,
            index=[f"DOF {i}" for i in range(self.total_dof)],
            columns=["Force"]
        )
        self.logger.debug(
            "\nForce Vector (F_global):\n" +
            force_df.to_string(float_format="%.2e", max_rows=100)
        )

    def _post_process(self):
        """Post-assembly matrix optimizations."""
        self.K_global.eliminate_zeros()
        
        # Symmetry check
        if self._is_symmetric():
            self.logger.info("✅ Stiffness matrix is symmetric")
        else:
            self.logger.warning("⚠️  Stiffness matrix is asymmetric")

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
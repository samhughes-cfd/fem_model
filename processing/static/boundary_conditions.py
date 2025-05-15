# processing/static/boundary_conditions.py

import numpy as np
import logging
import os
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix, diags, coo_matrix

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

def configure_boundary_logging(job_results_dir: str):
    """Configure hierarchical logging for boundary condition operations.
    
    Parameters
    ----------
    job_results_dir : str
        Output directory for boundary condition logs. Creates 'boundary_conditions.log'.
    """
    if job_results_dir:
        boundary_log_path = os.path.join(job_results_dir, "boundary_conditions.log")
        file_handler = logging.FileHandler(boundary_log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

def apply_boundary_conditions(K_global, F_global, job_results_dir: str = None):
    """Apply fixed boundary conditions with numerical stabilization and precision control.
    
    Implements the penalty method for constraint enforcement with automatic stabilization
    for ill-conditioned systems. Maintains float64 precision throughout all operations.

    Parameters
    ----------
    K_global : Union[scipy.sparse.csr_matrix, np.ndarray]
        Global stiffness matrix in sparse CSR or dense format. 
        Shape (n_dof, n_dof) where n_dof is total degrees of freedom.
    F_global : np.ndarray
        Global force vector. Shape (n_dof,) or (n_dof, 1).
    job_results_dir : str, optional
        Directory path for storing detailed boundary condition logs.

    Returns
    -------
    Tuple[scipy.sparse.csr_matrix, np.ndarray, np.ndarray]
        - K_mod: Modified stiffness matrix in CSR format with float64 precision
        - F_mod: Modified force vector with float64 precision (zeroed fixed DOFs)
        - fixed_dofs: Array of constrained degree of freedom indices

    Notes
    -----
    1. Uses LIL matrix format for efficient boundary condition application
    2. Implements automatic penalty scaling: penalty = 1e36 * max(diag(K))
    3. Adds numerical stabilization: K_stabilized = K + 1e-10*penalty*I
    4. Maintains float64 precision for all matrix operations
    5. Optimized logging handles both small and large systems

    Examples
    --------
    >>> from scipy.sparse import random
    >>> K = random(10, 10, density=0.5, format='csr', dtype=np.float64)
    >>> F = np.random.rand(10)
    >>> K_mod, F_mod, fixed_dofs = apply_boundary_conditions(K, F)
    """
    configure_boundary_logging(job_results_dir)
    logger.info("🔧 Applying stabilized boundary conditions...")

    # Precision-controlled initialization
    F_mod = np.asarray(F_global, dtype=np.float64).flatten()
    fixed_dofs = np.arange(6)

    # Matrix conversion with dtype enforcement
    if isinstance(K_global, csr_matrix):
        K_lil = K_global.astype(np.float64).tolil()
    else:
        K_lil = lil_matrix(K_global.astype(np.float64))

    # Dynamic penalty scaling
    max_diag = K_lil.diagonal().max() if K_lil.nnz else np.float64(1e36)
    penalty = np.float64(1e36) * max_diag

    # Boundary condition application
    K_lil[fixed_dofs, :] = 0.0
    K_lil[:, fixed_dofs] = 0.0
    for dof in fixed_dofs:
        K_lil[dof, dof] = penalty

    # Numerical stabilization
    stabilization = diags(
        [np.float64(1e-10) * penalty],
        [0],
        shape=K_lil.shape,
        format='lil',
        dtype=np.float64
    )
    K_stabilized = K_lil + stabilization

    # Final conversions with precision control
    K_mod = K_stabilized.tocsr().astype(np.float64)
    F_mod = F_mod.astype(np.float64)
    F_mod[fixed_dofs] = 0.0

    # System diagnostics
    if job_results_dir:
        _log_system_details(K_global, K_mod, F_mod)

    logger.info("✅ Boundary conditions applied with stabilization")
    return K_mod, F_mod, fixed_dofs

def _log_system_details(K_orig, K_mod, F_mod):
    """Log detailed system diagnostics with sparse matrix safety checks.
    
    Parameters
    ----------
    K_orig : Union[scipy.sparse.spmatrix, np.ndarray]
        Original stiffness matrix before BC application
    K_mod : scipy.sparse.csr_matrix
        Modified stiffness matrix after BC application
    F_mod : np.ndarray
        Modified force vector after BC application
    """
    try:
        file_logger = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
        
        # Matrix statistics
        orig_coo = K_orig.tocoo() if hasattr(K_orig, "tocoo") else coo_matrix(K_orig)
        mod_coo = K_mod.tocoo()
        
        file_logger.debug("\n=== System Diagnostics ===")
        file_logger.debug(f"Original DOFs: {orig_coo.shape[0]}")
        file_logger.debug(f"Original Non-zeros: {orig_coo.nnz}")
        file_logger.debug(f"Modified Non-zeros: {mod_coo.nnz}")
        file_logger.debug(f"Penalty Value: {K_mod.diagonal().max():.2e}")
        file_logger.debug(f"Stabilization Factor: {1e-10 * K_mod.diagonal().max():.2e}")

        # Sparse matrix logging
        if mod_coo.shape[0] <= 100:
            df_K = pd.DataFrame(mod_coo.toarray(), dtype=np.float64)
            file_logger.debug("\n🔍 Modified Stiffness Matrix:\n" + df_K.to_string(float_format="%.2e"))
        else:
            sparse_sample = pd.DataFrame({
                'Row': mod_coo.row,
                'Col': mod_coo.col,
                'Value': mod_coo.data
            }).sample(n=min(1000, len(mod_coo.data)))
            file_logger.debug("\n🔍 Matrix Sample (COO format):\n" + sparse_sample.to_string(index=False))

        # Condition number estimation
        diag_vals = mod_coo.diagonal()
        if np.any(diag_vals > 0):
            cond_estimate = diag_vals.max() / diag_vals[diag_vals > 0].min()
            file_logger.debug(f"\n⚠️ Condition Estimate: {cond_estimate:.1e}")

    except Exception as e:
        logger.error(f"Diagnostic logging failed: {str(e)}")
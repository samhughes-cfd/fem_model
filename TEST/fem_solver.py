import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from fem_logging import log_solver_metrics

def solve_sparse_system(K, F, output_dir, use_iterative=False):
    # Convert K to a sparse matrix for efficiency.
    K_sparse = sp.csc_matrix(K)
    start_time = time.time()
    
    if use_iterative:
        # Use the Conjugate Gradient iterative solver.
        solution, info = spla.cg(K_sparse, F, tol=1e-10)
    else:
        # Use a direct sparse solver.
        solution = spla.spsolve(K_sparse, F)
    
    solve_time = time.time() - start_time
    log_solver_metrics(K, F, solution, solve_time, output_dir)
    return solution
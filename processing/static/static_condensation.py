import numpy as np
from scipy.sparse import csr_matrix

def condensation(K_mod, F_mod, fixed_dofs, tol=1e-12):
    """
    Condenses the modified global stiffness matrix (K_mod) and force vector (F_mod)
    by removing:
    1) Explicitly fixed DOFs (primary condensation).
    2) Any remaining fully zero rows/columns (secondary condensation).

    Args:
        K_mod (csr_matrix): Stiffness matrix with boundary conditions applied.
        F_mod (np.ndarray): Force vector with boundary conditions applied.
        fixed_dofs (np.ndarray): Indices of fixed DOFs (from boundary conditions).
        tol (float, optional): Threshold below which matrix entries are considered zero. Defaults to 1e-12.

    Returns:
        Tuple[np.ndarray, np.ndarray, csr_matrix, np.ndarray]:
            - active_dofs: Indices of DOFs remaining after condensation.
            - inactive_dofs: Indices of DOFs that were pruned due to zero rows.
            - K_cond: The fully condensed global stiffness matrix.
            - F_cond: The fully condensed global force vector.
    """
    num_dofs = K_mod.shape[0]

    # PRIMARY CONDENSATION: Remove explicitly fixed DOFs (6 DOFs in this case)
    active_dofs = np.setdiff1d(np.arange(num_dofs), fixed_dofs)
    #print(f"Primary Condensation: {len(fixed_dofs)} fixed DOFs removed, {len(active_dofs)} remaining.")

    # Extract intermediate system where fixed DOFs have been removed
    K_intermediate = K_mod[active_dofs, :][:, active_dofs]
    F_intermediate = F_mod[active_dofs]

    # SECONDARY CONDENSATION: Identify and remove any fully zero rows/columns
    nonzero_rows = np.where(np.any(np.abs(K_intermediate.toarray()) > tol, axis=1))[0]
    fully_active_dofs = active_dofs[nonzero_rows]  # DOFs remaining after secondary condensation

    # Extract final condensed system
    K_cond = K_intermediate[nonzero_rows, :][:, nonzero_rows]
    F_cond = F_intermediate[nonzero_rows]

    # Identify DOFs that were pruned
    inactive_dofs = np.setdiff1d(active_dofs, fully_active_dofs)
    #print(f"Secondary Condensation: {len(inactive_dofs)} additional DOFs removed. Final DOFs: {len(fully_active_dofs)}")

    # SAFETY CHECK: Ensure non-trivial force remains after condensation
    if np.allclose(F_cond, 0, atol=tol):
        print("⚠️ Warning: Force vector is entirely zero after condensation! Check boundary conditions.")

    return fully_active_dofs, inactive_dofs, K_cond, F_cond


def reconstruction(active_dofs, U_cond, total_dofs):
    """
    Reconstructs the full displacement vector from the condensed solution.
    
    Parameters:
        active_dofs (ndarray): Indices of free DOFs in the condensed system.
        U_cond (ndarray): Displacement vector from solving the reduced system.
        total_dofs (int): Total number of DOFs in the original system.

    Returns:
        U_global (ndarray): Full displacement vector with zeros at constrained DOFs.
    """
    U_global = np.zeros(total_dofs)  # Initialize full displacement vector with zeros
    U_global[active_dofs] = U_cond  # Assign computed displacements to active DOFs
    return U_global
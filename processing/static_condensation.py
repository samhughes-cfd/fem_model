import numpy as np

def condensation(K_mod, F_mod, tol=1e-12, zero_inactive_forces=True):
    """
    Condenses the modified global stiffness matrix (K_mod) and force vector (F_mod)
    by removing rows and columns corresponding to inactive DOFs (i.e., rows where all
    entries are below a specified tolerance), along with the associated entries in F_mod.

    Args:
        K_mod (csr_matrix): 
            Global stiffness matrix with boundary conditions applied, in CSR format.
        F_mod (np.ndarray): 
            Global force vector with boundary conditions applied, as a 1D NumPy array.
        tol (float, optional): 
            Tolerance below which an entry is considered zero. Defaults to 1e-12.
        zero_inactive_forces (bool, optional): 
            If True, forces in inactive DOFs (those corresponding to zero rows in K_mod) are
            zeroed out. If False, a nonzero force in an inactive DOF will raise an error.
            Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray, csr_matrix, np.ndarray]:
            - active_dofs (np.ndarray): Indices of DOFs that are active (nonzero rows).
            - inactive_dofs (np.ndarray): Indices of DOFs that were pruned (all zero rows).
            - K_cond (csr_matrix): The condensed global stiffness matrix in CSR format.
            - F_cond (np.ndarray): The condensed global force vector.

    Raises:
        ValueError: If any pruned DOF has a nonzero force and zero_inactive_forces is False.
    """
    # Identify active DOFs: rows (and corresponding columns) with any entry above tol.
    # Conversion to a dense array is necessary to perform element-wise comparisons.
    active_dofs = np.where(np.any(np.abs(K_mod.toarray()) > tol, axis=1))[0]

    # Identify inactive DOFs.
    inactive_dofs = np.setdiff1d(np.arange(K_mod.shape[0]), active_dofs)

    # Check forces on inactive DOFs.
    if np.any(np.abs(F_mod[inactive_dofs]) > tol):
        if zero_inactive_forces:
            print("Warning: Inactive DOFs have nonzero forces. Zeroing out forces for DOFs:", inactive_dofs)
            F_mod[inactive_dofs] = 0.0
        else:
            raise ValueError("Force vector has nonzero entries for DOFs with zero stiffness. Inconsistent system!")

    # Build the reduced (condensed) system.
    # Slicing a CSR matrix with lists of indices returns a CSR matrix.
    K_cond = K_mod[active_dofs, :][:, active_dofs]
    F_cond = F_mod[active_dofs]

    return active_dofs, inactive_dofs, K_cond, F_cond

def reconstruction(active_dofs, U_cond, total_dofs):
    """
    Reconstructs the full displacement vector from the condensed solution.
    The pruned (inactive) DOFs are filled with zeros.

    Parameters:
        active_dofs (ndarray): Indices of DOFs used in the condensed system.
        U_cond (ndarray): Displacement vector from solving the condensed system.
        total_dofs (int): Total number of DOFs in the original system.

    Returns:
        U_global (ndarray): Full displacement vector with zeros inserted for inactive DOFs.
    """
    U_global = np.zeros(total_dofs)
    U_global[active_dofs] = U_cond
    return U_global
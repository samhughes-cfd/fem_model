import numpy as np

def static_condensation(K_mod, F_mod, tol=1e-12, zero_inactive_forces=True):
    """
    Condenses the modified global stiffness matrix K_mod and force vector F_mod
    by removing rows and columns of K_mod that are strictly zero, along with the 
    corresponding entries in F_mod.

    Parameters:
        K_mod (ndarray): Global stiffness matrix (with boundary conditions applied).
        F_mod (ndarray): Global force vector (with boundary conditions applied).
        tol (float): Tolerance below which an entry is considered zero.
        zero_inactive_forces (bool): If True, forces in inactive DOFs are zeroed out.
            If False, a nonzero force in an inactive DOF will raise an error.

    Returns:
        active_dofs (ndarray): Indices of DOFs that are active (nonzero rows).
        inactive_dofs (ndarray): Indices of DOFs that were pruned (all zero rows).
        K_condensed (ndarray): The condensed stiffness matrix.
        F_condensed (ndarray): The condensed force vector.

    Raises:
        ValueError: If any pruned DOF has a nonzero force and zero_inactive_forces is False.
    """
    # Identify active DOFs: rows (and corresponding columns) with any entry above tol.
    active_dofs = np.where(np.any(np.abs(K_mod) > tol, axis=1))[0]
    
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
    K_condensed = K_mod[np.ix_(active_dofs, active_dofs)]
    F_condensed = F_mod[active_dofs]
    
    return active_dofs, inactive_dofs, K_condensed, F_condensed

def reconstruct_full_solution(active_dofs, d_condensed, total_dofs):
    """
    Reconstructs the full displacement vector from the condensed solution.
    The pruned (inactive) DOFs are filled with zeros.

    Parameters:
        active_dofs (ndarray): Indices of DOFs used in the condensed system.
        d_condensed (ndarray): Displacement vector from solving the condensed system.
        total_dofs (int): Total number of DOFs in the original system.

    Returns:
        d_full (ndarray): Full displacement vector with zeros inserted for inactive DOFs.
    """
    d_full = np.zeros(total_dofs)
    d_full[active_dofs] = d_condensed
    return d_full
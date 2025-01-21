"""
processing/boundary_conditions.py

Applies a fixed boundary condition to the global stiffness matrix and force vector.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_boundary_conditions(K_global, F_global):
    """
    Apply a fixed [1 1 1 1 1 1] boundary condition to the system matrices.

    The first 6 DOFs (assumed to be at a single fixed node) are constrained 
    using the Penalty Method, enforcing zero displacements.

    Parameters
    ----------
    K_global : ndarray
        Global stiffness matrix (size: n x n).
    F_global : ndarray
        Global force vector (size: n x 1).

    Returns
    -------
    dict
        Contains the modified system matrices:
        - "K_mod": Modified stiffness matrix.
        - "F_mod": Modified force vector.

    Notes
    -----
    - Uses a **penalty value** to enforce the constraint.
    - Constraints are **hardcoded** to apply to the **first 6 DOFs**.
    - Suitable for **static** and **modal** analyses.
    """

    logger.info("Applying fixed [1 1 1 1 1 1] boundary condition.")

    # Define large penalty value
    large_penalty = 1e12  

    # Apply constraints to first 6 DOFs
    for dof in range(6):  
        K_global[dof, :] = 0
        K_global[:, dof] = 0
        K_global[dof, dof] = large_penalty
        F_global[dof, 0] = 0  # Enforce zero displacement

    logger.info("Fixed boundary conditions applied.")

    return {
        "K_mod": K_global,
        "F_mod": F_global,
    }
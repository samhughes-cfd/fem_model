"""
processing/boundary_conditions.py

Applies boundary conditions to the global stiffness matrix and force vector.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_boundary_conditions(K_global, F_global, constrained_dofs, analysis_type="static"):
    """
    Apply boundary conditions to the system matrices.

    Uses:
    - **Penalty Method** for static analysis.
    - **Mass/Boundary Reduction** for dynamic/modal analysis.

    Parameters:
        K_global (ndarray): Global stiffness matrix.
        F_global (ndarray): Global force vector.
        constrained_dofs (list): List of constrained DOFs (1D indices).
        analysis_type (str): Type of analysis ("static", "dynamic", "modal").

    Returns:
        dict: Contains the modified system matrices:
            - "K_mod": Modified stiffness matrix.
            - "F_mod": Modified force vector.
    """
    logger.info(f"Applying boundary conditions for {analysis_type} analysis.")

    K_mod = K_global.copy()
    F_mod = F_global.copy()

    large_penalty = 1e12  # Large stiffness to enforce constraints

    for dof in constrained_dofs:
        # Apply penalty method for static analysis
        K_mod[dof, :] = 0
        K_mod[:, dof] = 0
        K_mod[dof, dof] = large_penalty
        F_mod[dof, 0] = 0  # Zero displacement for static constraints

        # Commented out mass and damping modifications for future dynamic/modal tools
        # if M_mod is not None:
        #     M_mod[dof, :] = 0
        #     M_mod[:, dof] = 0
        #     M_mod[dof, dof] = 1  # Ensures stability of eigenvalue solutions

        # if C_mod is not None:
        #     C_mod[dof, :] = 0
        #     C_mod[:, dof] = 0

    logger.info(f"Boundary conditions applied for {analysis_type} analysis.")
    return {
        "K_mod": K_mod,
        # "M_mod": M_mod,
        # "C_mod": C_mod,
        "F_mod": F_mod,
    }
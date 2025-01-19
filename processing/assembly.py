"""
processing/assembly.py

Assembles the global stiffness matrix and force vector for static FEM problems.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def assemble_global_matrices(elements, stiffness_matrices, total_dof, analysis_type="static"):
    """
    Assembles the global system matrices (stiffness, mass, damping, force) based on the analysis type.

    Parameters:
        elements (list): List of instantiated element objects.
        stiffness_matrices (dict): Precomputed element stiffness matrices from `run_job.py`.
        total_dof (int): Total degrees of freedom.
        analysis_type (str): Type of analysis ("static", "dynamic", "modal").

    Returns:
        dict: Containing the assembled global matrices:
            - "K_global": Stiffness matrix
            - "F_global": Force vector (for static analysis)
            - "M_global": Mass matrix (if dynamic/modal, else None)
            - "C_global": Damping matrix (if dynamic, else None)
            - "element_stiffness_matrices": Dict of element stiffness matrices
    """
    logger.info(f"Assembling global system for {analysis_type} analysis.")

    # ✅ Initialize Global Matrices
    K_global = np.zeros((total_dof, total_dof))  # Stiffness matrix
    F_global = np.zeros(total_dof)  # Force vector (only used in static)

    # Commented out until dynamic/modal tools are available
    M_global = np.zeros((total_dof, total_dof)) if analysis_type in ["dynamic", "modal"] else None  # Mass matrix
    C_global = np.zeros((total_dof, total_dof)) if analysis_type == "dynamic" else None  # Damping matrix

    element_stiffness_matrices = {}

    # ✅ Loop through elements and assemble the global system
    for element in elements:
        try:
            Ke = stiffness_matrices[element.element_id]  # Retrieve precomputed Ke
            global_dof_indices = element.get_global_dof_indices()

            # Assemble stiffness matrix into global K
            K_global[np.ix_(global_dof_indices, global_dof_indices)] += Ke
            element_stiffness_matrices[element.element_id] = Ke

            # Assemble force vector (for static analysis)
            if analysis_type == "static":
                F_global[global_dof_indices] += element.Fe  # Accumulate element forces

            # Future expansion: Mass matrix assembly (disabled for now)
            # if M_global is not None and hasattr(element, "Me"):
            #     M_global[np.ix_(global_dof_indices, global_dof_indices)] += element.Me

            # Future expansion: Damping matrix assembly (disabled for now)
            # if C_global is not None and hasattr(element, "Ce"):
            #     C_global[np.ix_(global_dof_indices, global_dof_indices)] += element.Ce

            logger.info(f"  ✅ Element {element.element_id}: Added to K_global.")

        except Exception as e:
            logger.error(f"❌ Error assembling element {element.element_id}: {e}")

    logger.info(f"Global system assembly complete for {analysis_type} analysis.")

    return {
        "K_global": K_global,
        "F_global": F_global if analysis_type == "static" else None,
        "M_global": M_global,
        "C_global": C_global,
        "element_stiffness_matrices": element_stiffness_matrices,
    }
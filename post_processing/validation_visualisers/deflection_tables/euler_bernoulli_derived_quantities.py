# post_processing\validation_visualisers\deflection_tables\euler_bernoulli_derived_quantities.py

import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_derived_quantities(U_global, element_lengths, material_props, geometry_props):
    """
    Computes post-processed quantities: θ_z (nodal rotations), M(x) (internal bending moment),
    and V(x) (shear force) for Euler-Bernoulli beam elements.
    
    These are not modifications to the force vector but derived quantities for later analysis.

    Parameters:
    -----------
    U_global : np.ndarray
        Global displacement vector from FEM solve (DOFs ordered as [u_x, u_y, θ_z] per node).
    element_lengths : np.ndarray
        Array of element lengths.
    material_props : dict
        Contains material properties (E, G, etc.).
    geometry_props : dict
        Contains section properties (I_z, A, etc.).

    Returns:
    --------
    theta_z_values : np.ndarray
        Nodal rotations computed from u_y using finite differences.
    M_values : np.ndarray
        Internal bending moments computed from u_y using Euler-Bernoulli beam equation.
    V_values : np.ndarray
        Internal shear forces computed from bending moment derivative.
    """
    logger.info("Computing derived EB beam quantities: θ_z, M(x), and V(x)...")

    E = material_props["E"]
    I_z = geometry_props["I_z"]
    EI = E * I_z
    num_nodes = len(U_global) // 6  # 6 DOFs per node

    theta_z_values = np.zeros(num_nodes)
    M_values = np.zeros(num_nodes)
    V_values = np.zeros(num_nodes)

    # Compute θ_z (rotation)
    for i in range(num_nodes):
        index = 6 * i + 1  # u_y index
        L = element_lengths[i] if i < len(element_lengths) else element_lengths[-1]
        
        if i == 0:
            theta_z_values[i] = (-3 * U_global[index] + 4 * U_global[index + 6] - U_global[index + 12]) / (2 * L)
        elif i == num_nodes - 1:
            theta_z_values[i] = (3 * U_global[index] - 4 * U_global[index - 6] + U_global[index - 12]) / (2 * L)
        else:
            theta_z_values[i] = (U_global[index + 6] - U_global[index]) / L

    logger.debug("Computed θ_z values: %s", theta_z_values)

    # Compute M(x) (Bending Moment)
    for i in range(num_nodes):
        index = 6 * i + 1
        L = element_lengths[i] if i < len(element_lengths) else element_lengths[-1]

        if i == 0:
            M_values[i] = EI * (2 * U_global[index] - 5 * U_global[index + 6] + 4 * U_global[index + 12] - U_global[index + 18]) / (L**2)
        elif i == num_nodes - 1:
            M_values[i] = EI * (2 * U_global[index] - 5 * U_global[index - 6] + 4 * U_global[index - 12] - U_global[index - 18]) / (L**2)
        else:
            M_values[i] = EI * (U_global[index + 6] - 2 * U_global[index] + U_global[index - 6]) / (L**2)

    logger.debug("Computed M(x) values: %s", M_values)

    # Compute V(x) (Shear Force) as the derivative of M(x)
    for i in range(num_nodes):
        L = element_lengths[i] if i < len(element_lengths) else element_lengths[-1]

        if i == 0:
            V_values[i] = (M_values[i + 1] - M_values[i]) / L
        elif i == num_nodes - 1:
            V_values[i] = (M_values[i] - M_values[i - 1]) / L
        else:
            V_values[i] = (M_values[i + 1] - M_values[i - 1]) / (2 * L)

    logger.debug("Computed V(x) values: %s", V_values)

    return theta_z_values, M_values, V_values

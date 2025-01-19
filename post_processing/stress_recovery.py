# post_processing/stress_recovery.py

import numpy as np
from pre_processing.element_library.euler_bernoulli.euler_bernoulli import EulerBernoulliBeamElement
from pre_processing.element_library.timoshenko.timoshenko import TimoshenkoBeamElement

def compute_stresses(elements_list, displacements):
    """
    Compute stresses for all elements in the elements_list.

    Parameters:
        elements_list (list): List of beam element instances.
        displacements (ndarray): Global displacement vector.

    Returns:
        dict: Dictionary containing lists of axial, bending, and shear stresses.
    """
    axial_stress = []
    bending_stress = []
    shear_stress = []
    element_centers = []

    for element in elements_list:
        # Get global DOF indices for the element
        dof_indices = element.get_global_dof_indices()
        u_elem = displacements[dof_indices]

        # Compute stresses based on element type
        if isinstance(element, EulerBernoulliBeamElement):
            stresses = compute_euler_bernoulli_stress(element, u_elem)
        elif isinstance(element, TimoshenkoBeamElement):
            stresses = compute_timoshenko_stress(element, u_elem)
        else:
            raise TypeError(f"Unknown element type: {type(element)}")

        axial_stress.append(stresses['axial_stress'])
        bending_stress.append(stresses['bending_stress'])
        shear_stress.append(stresses['shear_stress'])

        # Compute element center for plotting
        node_ids = element.geometry.elements[element.element_id]
        x1 = element.geometry.node_positions[node_ids[0]]
        x2 = element.geometry.node_positions[node_ids[1]]
        element_center = (x1 + x2) / 2
        element_centers.append(element_center)

    return {
        'axial_stress': np.array(axial_stress),
        'bending_stress': np.array(bending_stress),
        'shear_stress': np.array(shear_stress),
        'element_centers': np.array(element_centers)
    }

def compute_euler_bernoulli_stress(element, u_elem):
    """
    Compute axial and bending stress for the Euler-Bernoulli beam element.

    Parameters:
        element (EulerBernoulliBeamElement): The beam element instance.
        u_elem (ndarray): Element displacement vector.

    Returns:
        dict: Dictionary containing axial stress, bending stress, and shear stress.
    """
    element_length = element.geometry.get_element_length(element.element_id)
    E = element.material.E
    # Axial strain
    axial_strain = (u_elem[3] - u_elem[0]) / element_length
    axial_stress = E * axial_strain

    # Curvature
    curvature = (u_elem[5] - u_elem[2]) / element_length
    c = element.section_height / 2  # Distance from neutral axis to outer fiber
    bending_stress = E * curvature * c

    # Shear stress (zero for Euler-Bernoulli beams)
    shear_stress = 0.0

    return {
        'axial_stress': axial_stress,
        'bending_stress': bending_stress,
        'shear_stress': shear_stress
    }

def compute_timoshenko_stress(element, u_elem):
    """
    Compute axial, bending, and shear stress for the Timoshenko beam element.

    Parameters:
        element (TimoshenkoBeamElement): The beam element instance.
        u_elem (ndarray): Element displacement vector.

    Returns:
        dict: Dictionary containing axial stress, bending stress, and shear stress.
    """
    element_length = element.geometry.get_element_length(element.element_id)
    E = element.material.E
    G = element.material.G
    ks = element.ks
    # Axial strain
    axial_strain = (u_elem[3] - u_elem[0]) / element_length
    axial_stress = E * axial_strain

    # Bending curvature
    curvature = (u_elem[5] - u_elem[2]) / element_length
    c = element.section_height / 2  # Distance from neutral axis to outer fiber
    bending_stress = E * curvature * c

    # Shear strain
    shear_strain = ((u_elem[4] - u_elem[1]) / element_length) - ((u_elem[2] + u_elem[5]) / 2)
    shear_stress = ks * G * shear_strain

    return {
        'axial_stress': axial_stress,
        'bending_stress': bending_stress,
        'shear_stress': shear_stress
    }
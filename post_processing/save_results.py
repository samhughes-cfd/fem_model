# post_processing/save_results.py

import numpy as np
import logging
import os
import pandas as pd

def save_displacements(displacement_results, element_type_dir, element_name, timestamp):
    """
    Save nodal displacements and rotations to a .txt file in a matrix format.

    Parameters:
        displacement_results (dict): Dictionary containing displacement arrays and node positions.
        element_type_dir (str): Directory where the results for the current element type are stored.
        element_name (str): Name of the element type (e.g., "EulerBernoulliBeamElement").
        timestamp (str): Timestamp string to append to the filename.
    """
    os.makedirs(element_type_dir, exist_ok=True)
    save_path = os.path.join(element_type_dir, f'displacements_{element_name}_{timestamp}.txt')

    # Extract data from the dictionary
    node_positions = displacement_results['node_positions']
    u = displacement_results['u']
    w = displacement_results['w']
    theta = displacement_results['theta']

    # Stack displacements into a 3 x n matrix
    displacement_matrix = np.vstack((u, w, theta))

    # Create labels for nodes and degrees of freedom
    node_labels = [f"Node {i}" for i in range(len(u))]
    dof_labels = ['u (m)', 'w (m)', 'theta (rad)']

    # Create a DataFrame for displacements
    displacement_df = pd.DataFrame(
        displacement_matrix,
        index=dof_labels,
        columns=node_labels
    )

    # Include node positions as a row
    node_positions_row = pd.Series(node_positions, index=node_labels, name='Position (m)')

    # Transpose the DataFrame so nodes are rows
    displacement_df = displacement_df.transpose()

    # Insert node positions as the first column
    displacement_df.insert(0, 'Position (m)', node_positions_row)

    # Save the DataFrame to a text file
    try:
        with open(save_path, 'w') as f:
            f.write(f"Nodal Displacements and Rotations for {element_name} Elements:\n")
            f.write(displacement_df.to_string(float_format='{:.6e}'.format))
        logging.info(f"Displacements saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving displacements: {e}")


def save_stresses(stress_results, element_type_dir, element_name, timestamp):
    """
    Save stress results to a .txt file in a matrix format using a DataFrame.

    Parameters:
        stress_results (dict): Dictionary containing stress arrays and element centers.
        element_type_dir (str): Directory where the results for the current element type are stored.
        element_name (str): Name of the element type (e.g., "EulerBernoulliBeamElement").
        timestamp (str): Timestamp string to append to the filename.
    """
    os.makedirs(element_type_dir, exist_ok=True)
    save_path = os.path.join(element_type_dir, f'stresses_{element_name}_{timestamp}.txt')

    element_centers = stress_results['element_centers']
    axial_stress = stress_results['axial_stress']
    bending_stress = stress_results['bending_stress']
    shear_stress = stress_results['shear_stress']

    # Create labels for elements
    element_labels = [f"Element {i}" for i in range(len(element_centers))]

    # Create a DataFrame for stresses
    data = {
        'Element Center (m)': element_centers,
        'Axial Stress (Pa)': axial_stress,
        'Bending Stress (Pa)': bending_stress,
        'Shear Stress (Pa)': shear_stress
    }
    stress_df = pd.DataFrame(data, index=element_labels)

    # Save the DataFrame to a text file
    try:
        with open(save_path, 'w') as f:
            f.write(f"Stress Results for {element_name} Elements:\n")
            f.write(stress_df.to_string(float_format='{:.6e}'.format))
        logging.info(f"Stresses saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving stresses: {e}")
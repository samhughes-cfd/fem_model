# post_processing/load_visualisation.py

import matplotlib.pyplot as plt
import os
import logging

def plot_loads(distributed_loads, point_loads, node_positions, element_type_dir, timestamp):
    """
    Generates a loads visualization plot and saves it.

    Parameters:
        distributed_loads (dict): Distributed loads applied.
        point_loads (dict): Point loads applied.
        node_positions (list): Positions of nodes.
        element_type_dir (str): Directory where the results for the current element type are stored.
        timestamp (str): Timestamp string to append to the filename.
    """
    try:
        num_elements = len(node_positions) - 1
        x_elements = []
        q_values = []

        for elem_id in range(num_elements):
            x1 = node_positions[elem_id]
            x2 = node_positions[elem_id + 1]
            x_elem_center = (x1 + x2) / 2.0
            x_elements.append(x_elem_center)
            q = distributed_loads.get(elem_id, {}).get('q', 0.0)  # Updated from 'qy' to 'q'
            q_values.append(q)

        plt.figure(figsize=(10, 6))
        plt.plot(x_elements, q_values, label='Distributed Load (q)', color='blue', linestyle='-', linewidth=2)

        # Plot point loads
        for node_id, loads in point_loads.items():
            x = node_positions[node_id]
            q = loads.get('w', 0.0)  # Assuming 'w' is the vertical point load
            if q != 0.0:
                plt.plot(x, q, 'ro', label='Point Load' if node_id == list(point_loads.keys())[0] else "")

        plt.xlabel('Position along the beam (m)')
        plt.ylabel('Load (N/m or N)')
        plt.title('Beam Loads Visualization')
        plt.legend()
        plt.grid(True)

        # Define the save path with timestamp
        save_path = os.path.join(element_type_dir, f"loads_{timestamp}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Loads visualization plot saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to generate loads visualization plot: {e}")

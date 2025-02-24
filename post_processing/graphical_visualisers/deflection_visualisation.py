# post_processing/deflection_visualisation.py

import matplotlib.pyplot as plt
import os
import logging

def plot_deflection_comparison(results, element_type_dir, timestamp):
    """
    Generates a deflection comparison plot and saves it.

    Parameters:
        results (dict): Simulation results.
        element_type_dir (str): Directory where the results for the current element type are stored.
        timestamp (str): Timestamp string to append to the filename.
    """
    try:
        plt.figure(figsize=(10, 6))
        for beam_name, data in results.items():
            x = data['node_positions']
            w = data['w']
            w_mm = w * 1000  # Convert deflection from meters to millimeters
            plt.plot(x, w_mm, label=f'{beam_name}')
        plt.xlabel('Position along the beam (m)')
        plt.ylabel('Deflection (mm)')  # Updated to millimeters
        plt.title('Transverse Deflection Comparison')
        plt.legend()
        plt.grid(True)
        
        # Define the save path with timestamp
        save_path = os.path.join(element_type_dir, f'deflection_comparison_{timestamp}.png')
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Deflection comparison plot saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to generate deflection comparison plot: {e}")
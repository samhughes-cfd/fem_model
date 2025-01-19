# post_processing/stress_visualisation.py

import matplotlib.pyplot as plt
import os
import logging

def plot_stress_comparison(results, stress_type='bending_stress', element_type_dir=None, timestamp=None):
    """
    Generates a stress comparison plot and saves it.

    Parameters:
        results (dict): Simulation results.
        stress_type (str): Type of stress to plot (e.g., 'bending_stress').
        element_type_dir (str): Directory where the results for the current element type are stored.
        timestamp (str): Timestamp string to append to the filename.
    """
    try:
        plt.figure(figsize=(10, 6))
        for beam_name, data in results.items():
            x_elements = data['element_centers']
            stress = data[stress_type]
            stress_MPa = stress / 1e6  # Convert stress from Pa to MPa
            plt.plot(x_elements, stress_MPa, label=f'{beam_name}')
        plt.xlabel('Position along the beam (m)')
        plt.ylabel('Stress (MPa)')  # Updated to MPa
        plt.title(f'{stress_type.replace("_", " ").title()} Comparison')
        plt.legend()
        plt.grid(True)
        
        # Format y-axis labels to plain numbers with two decimal places
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        
        # Adjust layout to prevent cutting off labels
        plt.tight_layout()
        
        # Define the save path with timestamp
        save_path = os.path.join(element_type_dir, f"{stress_type}_comparison_{timestamp}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logging.info(f"{stress_type.replace('_', ' ').title()} comparison plot saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to generate {stress_type.replace('_', ' ').title()} comparison plot: {e}")

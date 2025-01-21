# load_library/schemes/visualize_load.py

import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import argparse
from datetime import datetime

# ============================
# Global Input Variables
# ============================

# Path to the load profile .txt file
LOAD_PROFILE_PATH = '../../load_library/load_profiles/load_profile_latest.txt'  # Placeholder path

# Directory to save the plot images
SAVE_DIRECTORY = '../../load_library/load_profiles/'

# ============================

def read_load_profile(profile_file_path):
    """
    Read the load profile TXT file.

    Parameters:
    - profile_file_path (str): Path to the load profile TXT file.

    Returns:
    - pd.DataFrame: DataFrame containing [x], [y], [z], [F_x], [F_y], [F_z], [M_x], [M_y], [M_z].
    """
    try:
        # Read the file with headers in square brackets
        load_profile_df = pd.read_csv(profile_file_path, delim_whitespace=True)
        return load_profile_df
    except Exception as e:
        logging.error(f"Error reading load profile file: {e}")
        raise

def visualize_load_distribution(load_profile_df, show_plots=True, save_plots=False, save_directory='../../load_library/load_profiles/'):
    """
    Visualize the load distribution along the beam.

    Parameters:
    - load_profile_df (pd.DataFrame): DataFrame containing load profile data.
    - show_plots (bool): Whether to display the plots.
    - save_plots (bool): Whether to save the plots as image files.
    - save_directory (str): Directory to save the plot images if save_plots is True.
    """
    os.makedirs(save_directory, exist_ok=True)
    
    # Create a figure with two subplots: Forces and Moments
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot Forces
    axes[0].plot(load_profile_df['[x]'], load_profile_df['[F_x]'], label='F_x', marker='o')
    axes[0].plot(load_profile_df['[x]'], load_profile_df['[F_y]'], label='F_y', marker='o')
    axes[0].plot(load_profile_df['[x]'], load_profile_df['[F_z]'], label='F_z', marker='o')
    axes[0].set_title('Force Distribution Along the Beam')
    axes[0].set_ylabel('Force (N)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Moments
    axes[1].plot(load_profile_df['[x]'], load_profile_df['[M_x]'], label='M_x', marker='o')
    axes[1].plot(load_profile_df['[x]'], load_profile_df['[M_y]'], label='M_y', marker='o')
    axes[1].plot(load_profile_df['[x]'], load_profile_df['[M_z]'], label='M_z', marker='o')
    axes[1].set_title('Moment Distribution Along the Beam')
    axes[1].set_xlabel('x (m)')
    axes[1].set_ylabel('Moment (Nm)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        forces_plot = os.path.join(save_directory, f"forces_plot_{timestamp}.png")
        moments_plot = os.path.join(save_directory, f"moments_plot_{timestamp}.png")
        fig.savefig(forces_plot)
        fig.savefig(moments_plot)
        logging.info(f"Plots saved to '{forces_plot}' and '{moments_plot}'.")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def main():
    """
    Main function to visualize the load profile.
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Visualize Load Profiles.')
    parser.add_argument('--profile', type=str, default=LOAD_PROFILE_PATH,
                        help=f'Path to the load profile TXT file. Default: {LOAD_PROFILE_PATH}')
    parser.add_argument('--show', action='store_true', help='Show the plots.')
    parser.add_argument('--save', action='store_true', help='Save the plots as image files.')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIRECTORY,
                        help=f'Directory to save plot images. Default: {SAVE_DIRECTORY}')
    
    args = parser.parse_args()
    
    # Update global variables if command-line arguments are provided
    global LOAD_PROFILE_PATH, SAVE_DIRECTORY
    LOAD_PROFILE_PATH = args.profile
    SAVE_DIRECTORY = args.save_dir
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        # Validate profile file path
        if not os.path.isfile(LOAD_PROFILE_PATH):
            logging.error(f"The specified load profile file does not exist: {LOAD_PROFILE_PATH}")
            return
        
        # Read the load profile
        load_profile_df = read_load_profile(LOAD_PROFILE_PATH)
        logging.info(f"Load profile '{LOAD_PROFILE_PATH}' read successfully.")
        
        # Visualize the load distribution
        visualize_load_distribution(load_profile_df, show_plots=args.show, save_plots=args.save, save_directory=SAVE_DIRECTORY)
        
        logging.info("Load visualization completed successfully.")
        
    except Exception as e:
        logging.error(f"An error occurred during load visualization: {e}")

if __name__ == "__main__":
    main()
# load_library/schemes/generate_load.py

import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime
import argparse

def read_load_data(dat_file_path, beam_length):
    """
    Read load data from a .dat file and convert r/R to x coordinates.

    Parameters:
    - dat_file_path (str): Path to the .dat load data file.
    - beam_length (float): Length of the beam (L) in meters.

    Returns:
    - pd.DataFrame: DataFrame containing x, Fx, Fy, Fz, Mx, My, Mz.
    """
    # Define column names based on provided data
    column_names = ['r_R', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    
    try:
        # Read the data using pandas
        load_df = pd.read_csv(
            dat_file_path, 
            delim_whitespace=True, 
            header=None, 
            names=column_names,
            comment='#'  # Assuming '#' is used for comments
        )
    except Exception as e:
        logging.error(f"Error reading load data file '{dat_file_path}': {e}")
        raise
    
    # Convert r/R to x using x = (r/R) * L
    load_df['x'] = load_df['r_R'] * beam_length
    
    # Validate that x does not exceed beam length
    if load_df['x'].max() > beam_length:
        logging.error("Load position exceeds beam length.")
        raise ValueError("Load position exceeds beam length.")
    
    # Rearrange columns to have x first
    load_df = load_df[['x', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']]
    
    return load_df

def generate_load_profile(load_df, save_directory):
    """
    Generate the load profile file with spatial coordinates and load components.

    Parameters:
    - load_df (pd.DataFrame): DataFrame containing load data with x positions.
    - save_directory (str): Directory to save the load profile file.
    """
    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)
    
    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"load_profile_{timestamp}.txt"
    filepath = os.path.join(save_directory, filename)
    
    # Define column headers with square brackets
    headers = ['[x]', '[y]', '[z]', '[F_x]', '[F_y]', '[F_z]', '[M_x]', '[M_y]', '[M_z]']
    
    # Define fixed widths for each column (in characters)
    # Adjust widths based on expected data ranges
    column_widths = {
        '[x]': 12,
        '[y]': 12,
        '[z]': 12,
        '[F_x]': 14,
        '[F_y]': 14,
        '[F_z]': 14,
        '[M_x]': 14,
        '[M_y]': 14,
        '[M_z]': 14
    }
    
    # Create format string for headers and rows
    header_fmt = ''.join([f"{{:<{column_widths[col]}}}" for col in headers])
    row_fmt = ''.join([f"{{:<{column_widths[col]}}}" for col in headers])
    
    # Open the file for writing
    try:
        with open(filepath, 'w') as f:
            # Write the header
            header_line = header_fmt.format(*headers)
            f.write(header_line + '\n')
            
            # Write each row with formatted values
            for _, row in load_df.iterrows():
                x = f"{row['x']:.6f}"
                y = "0.0"  # Assuming y=0.0 for all loads
                z = "0.0"  # Assuming z=0.0 for all loads
                F_x = f"{row['Fx']:.6f}"
                F_y = f"{row['Fy']:.6f}"
                F_z = f"{row['Fz']:.6f}"
                M_x = f"{row['Mx']:.6f}"
                M_y = f"{row['My']:.6f}"
                M_z = f"{row['Mz']:.6f}"
                
                # Prepare the row data
                row_data = [x, y, z, F_x, F_y, F_z, M_x, M_y, M_z]
                
                # Format the row with fixed widths
                row_line = row_fmt.format(*row_data)
                f.write(row_line + '\n')
        
        logging.info(f"Load profile file saved to '{filepath}'.")
    except Exception as e:
        logging.error(f"Error saving load profile file '{filepath}': {e}")
        raise

def main():
    """
    Main function to generate and save the load profile file.
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Generate Load Profiles from Load Data.')
    parser.add_argument('--load_data', type=str, default='pre_processing/load_library/data/tidal_benchmark/Force_distribution_TSR4.dat',
                        help='Path to loads.dat file. Default: pre_processing/load_library/data/tidal_benchmark/Force_distribution_TSR4.dat')
    parser.add_argument('--beam_length', type=float, default=8.0,
                        help='Beam length (L) in meters. Default: 8.0')
    parser.add_argument('--save_dir', type=str, default='pre_processing/load_library/load_profiles/tidal_benchmark',
                        help='Directory to save load profiles. Default: pre_processing/load_library/load_profiles/tidal_benchmark')
    
    args = parser.parse_args()
    
    # Assign variables from arguments
    load_data_path = args.load_data
    beam_length = args.beam_length
    save_directory = args.save_dir
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        # Validate load_data path
        if not os.path.isfile(load_data_path):
            logging.error(f"The specified load data file does not exist: {load_data_path}")
            return
        
        # Read load data
        load_df = read_load_data(load_data_path, beam_length=beam_length)
        logging.info(f"Load data read from '{load_data_path}' with {len(load_df)} entries.")
        
        # Generate load profile
        generate_load_profile(load_df, save_directory=save_directory)
        
        logging.info("Load generation completed successfully.")
        
    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {fnf_error}")
    except ValueError as ve:
        logging.error(f"Value error: {ve}")
    except Exception as e:
        logging.error(f"An error occurred during load generation: {e}")

if __name__ == "__main__":
    main()
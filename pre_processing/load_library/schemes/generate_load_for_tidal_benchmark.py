# pre_processing\load_library\schemes\generate_load_for_tidal_benchmark.py

import numpy as np
import pandas as pd
import logging
import os
import argparse
import uuid
from datetime import datetime
import re
import sys

def setup_logging(log_level, logging_dir, tsrx_name, timestamp):
    """
    Configures the logging settings.

    Parameters:
    - log_level (str): Logging level as a string (e.g., 'DEBUG', 'INFO').
    - logging_dir (str): Directory where log files will be stored.
    - tsrx_name (str): The TSRX identifier extracted from the source file.
    - timestamp (str): The current timestamp formatted as YYYYMMDD_HHMMSS.
    """
    try:
        # Ensure the logging directory exists
        os.makedirs(logging_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create logging directory '{logging_dir}': {e}")
        sys.exit(1)
    
    # Define the log filename with TSRX name and timestamp
    log_filename = f"log_{tsrx_name}_{timestamp}.log"
    log_filepath = os.path.join(logging_dir, log_filename)
    
    # Create a logger
    logger = logging.getLogger(tsrx_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Define log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler for logging
    try:
        file_handler = logging.FileHandler(log_filepath)
    except Exception as e:
        print(f"Failed to create log file '{log_filepath}': {e}")
        sys.exit(1)
        
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create console handler for logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: '{log_filepath}'")
    
    return logger

def extract_tsrx_name(dat_file_path):
    """
    Extracts the TSRX name from the source .dat filename.

    Parameters:
    - dat_file_path (str): Path to the .dat load data file.

    Returns:
    - str: Extracted TSRX name.

    Raises:
    - ValueError: If TSRX name cannot be extracted.
    """
    # Example: Force_distribution_TSR4.dat -> TSR4
    basename = os.path.basename(dat_file_path)
    match = re.search(r'TSR\d+', basename, re.IGNORECASE)
    if match:
        tsrx_name = match.group(0).upper()
        return tsrx_name
    else:
        raise ValueError(f"TSRX name could not be extracted from filename: '{basename}'")

def read_load_data(dat_file_path, logger):
    """
    Reads load data from a .dat file.

    Parameters:
    - dat_file_path (str): Path to the .dat load data file.
    - logger (logging.Logger): Logger for logging messages.

    Returns:
    - pd.DataFrame: DataFrame containing r/R, Fx, Fy, Fz, Mx, My, Mz.

    Raises:
    - FileNotFoundError: If the .dat file does not exist.
    - ValueError: If required columns are missing.
    """
    column_names = ['r_R', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    
    try:
        load_df = pd.read_csv(
            dat_file_path, 
            sep='\s+', 
            header=None, 
            names=column_names,
            comment='#'  # Assuming '#' is used for comments
        )
        logger.info(f"Successfully read load data from '{dat_file_path}'.")
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {fnf_error}")
        raise
    except pd.errors.ParserError as parse_error:
        logger.error(f"Parsing error while reading '{dat_file_path}': {parse_error}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while reading '{dat_file_path}': {e}")
        raise
    
    # Validate required columns
    required_columns = ['r_R', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    if not all(col in load_df.columns for col in required_columns):
        error_msg = "Input data is missing required columns."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if load_df.empty:
        error_msg = "Input data is empty."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Load data contains {len(load_df)} entries.")
    return load_df

def denormalize_loads(load_df, R, logger):
    """
    Denormalizes positions and load components.

    Parameters:
    - load_df (pd.DataFrame): DataFrame with normalized data.
    - R (float): Normalization factor for denormalizing r/R.
    - logger (logging.Logger): Logger for logging messages.

    Returns:
    - pd.DataFrame: Denormalized DataFrame with 'x' positions and load components.

    Raises:
    - Exception: If any error occurs during denormalization.
    """
    try:
        load_df_denormalized = load_df.copy()
        load_df_denormalized['x'] = load_df_denormalized['r_R'] * R  # Denormalize position
        logger.info("Position denormalization completed.")
        
        # Denormalize load components by multiplying with 'x'
        load_df_denormalized['Fx'] = load_df_denormalized['Fx'] * load_df_denormalized['x']  # [N/m] * [m] = [N]
        load_df_denormalized['Fy'] = load_df_denormalized['Fy'] * load_df_denormalized['x']
        load_df_denormalized['Fz'] = load_df_denormalized['Fz'] * load_df_denormalized['x']
        load_df_denormalized['Mx'] = load_df_denormalized['Mx'] * load_df_denormalized['x']  # [N] * [m] = [N·m]
        load_df_denormalized['My'] = load_df_denormalized['My'] * load_df_denormalized['x']
        load_df_denormalized['Mz'] = load_df_denormalized['Mz'] * load_df_denormalized['x']
        logger.info("Load components denormalization completed.")
        
    except Exception as e:
        logger.error(f"Error during denormalization: {e}")
        raise
    
    return load_df_denormalized

def shift_load_profile(load_df_denormalized, logger):
    """
    Shifts the load profile so that the first load is at x = 0.0 m.

    Parameters:
    - load_df_denormalized (pd.DataFrame): Denormalized DataFrame with 'x' positions.
    - logger (logging.Logger): Logger for logging messages.

    Returns:
    - pd.DataFrame: Shifted DataFrame.

    Raises:
    - Exception: If any error occurs during shifting.
    """
    try:
        if load_df_denormalized.empty:
            error_msg = "Denormalized load data is empty. Cannot perform shifting."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        x_shift = load_df_denormalized['x'].iloc[0]
        logger.info(f"Shifting all x positions by {x_shift:.6f} meters.")
        load_df_denormalized['x'] = load_df_denormalized['x'] - x_shift
        logger.info("Load profile shifting completed.")
    except Exception as e:
        logger.error(f"Error during load profile shifting: {e}")
        raise
    
    return load_df_denormalized

def map_load_components(load_df_final, logger):
    """
    Maps load components from source to target coordinate system.

    Parameters:
    - load_df_final (pd.DataFrame): DataFrame with denormalized and shifted load components.
    - logger (logging.Logger): Logger for logging messages.

    Returns:
    - pd.DataFrame: DataFrame with mapped load components.
    """
    load_mapping = {
        'Fx': 'Fy',  # Fx (source) → Fy (target)
        'Fy': 'Fz',  # Fy → Fz
        'Fz': 'Fx',  # Fz → Fx
        'Mx': 'My',  # Mx → My
        'My': 'Mz',  # My → Mz
        'Mz': 'Mx'   # Mz → Mx
    }
    
    try:
        load_df_mapped = load_df_final.rename(columns=load_mapping)
        logger.info("Load components mapping completed.")
    except Exception as e:
        logger.error(f"Error during load components mapping: {e}")
        raise
    
    return load_df_mapped

def generate_load_profile(load_df, save_directory, tsrx_name, timestamp, logger):
    """
    Generates the load profile file with spatial coordinates and load components.

    Parameters:
    - load_df (pd.DataFrame): DataFrame containing load data with denormalized and mapped load components.
    - save_directory (str): Directory to save the load profile file.
    - tsrx_name (str): The TSRX identifier extracted from the source file.
    - timestamp (str): The current timestamp formatted as YYYYMMDD_HHMMSS.
    - logger (logging.Logger): Logger for logging messages.

    Raises:
    - IOError: If there's an issue writing to the file.
    """
    # Define column headers with square brackets
    headers = ['[x]', '[y]', '[z]', '[F_x]', '[F_y]', '[F_z]', '[M_x]', '[M_y]', '[M_z]']
    
    # Define fixed widths for each column (in characters)
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
    
    # Define format strings for each column based on fixed widths
    format_strings = {
        '[x]': f"{{:<{column_widths['[x]']}.6f}}",
        '[y]': f"{{:<{column_widths['[y]']}.6f}}",
        '[z]': f"{{:<{column_widths['[z]']}.6f}}",
        '[F_x]': f"{{:<{column_widths['[F_x]']}.6f}}",
        '[F_y]': f"{{:<{column_widths['[F_y]']}.6f}}",
        '[F_z]': f"{{:<{column_widths['[F_z]']}.6f}}",
        '[M_x]': f"{{:<{column_widths['[M_x]']}.6f}}",
        '[M_y]': f"{{:<{column_widths['[M_y]']}.6f}}",
        '[M_z]': f"{{:<{column_widths['[M_z]']}.6f}}"
    }
    
    # Define a mapping from column names to header names
    column_to_header = {
        'x': '[x]',
        'y': '[y]',
        'z': '[z]',
        'Fx': '[F_x]',
        'Fy': '[F_y]',
        'Fz': '[F_z]',
        'Mx': '[M_x]',
        'My': '[M_y]',
        'Mz': '[M_z]'
    }
    
    try:
        # Add 'y' and 'z' columns with default value 0.0 if not present
        if 'y' not in load_df.columns:
            load_df['y'] = 0.0
            logger.debug("Added 'y' column with default value 0.0.")
        if 'z' not in load_df.columns:
            load_df['z'] = 0.0
            logger.debug("Added 'z' column with default value 0.0.")
        
        # Reorder columns to match the headers
        load_df_final = load_df[['x', 'y', 'z', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']]
        logger.debug("Reordered DataFrame columns to match headers.")
        
        # Apply formatting to each column using vectorized operations
        for col in load_df_final.columns:
            header = column_to_header.get(col)
            if header:
                try:
                    # Convert to float and format
                    load_df_final[col] = load_df_final[col].astype(float).map(lambda x: format_strings[header].format(x))
                    logger.debug(f"Formatted column '{col}' with header '{header}'.")
                except ValueError as ve:
                    logger.error(f"Value error in column '{col}': {ve}")
                    raise
            else:
                logger.warning(f"No header mapping defined for column '{col}'. Skipping formatting.")
        
        # Concatenate all formatted columns into a single string per row
        load_df_final['formatted_row'] = load_df_final.apply(
            lambda row: ''.join([row[col] for col in ['x', 'y', 'z', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']]),
            axis=1
        )
        logger.debug("Concatenated formatted columns into 'formatted_row'.")
        
        # Define the header line with fixed widths
        header_line = ''.join([f"{col:<{column_widths[col]}}" for col in headers])
        
        # Define the load profile filename with TSRX name and timestamp
        load_profile_filename = f"load_profile_{tsrx_name}_{timestamp}.txt"
        load_profile_filepath = os.path.join(save_directory, load_profile_filename)
        
        # Write to file
        with open(load_profile_filepath, 'w') as f:
            # Write the header
            f.write(header_line + '\n')
            logger.debug("Written header to the output file.")
            
            # Write all formatted rows at once
            f.write('\n'.join(load_df_final['formatted_row']) + '\n')
            logger.debug(f"Written {len(load_df_final)} formatted rows to the output file.")
        
        logger.info(f"Load profile file successfully saved to '{load_profile_filepath}'.")
    
    except IOError as io_error:
        logger.error(f"I/O error while writing to file '{load_profile_filepath}': {io_error}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during load profile generation: {e}")
        raise

def process_dat_file(dat_file_path, beam_length, R, save_dir, logging_dir, log_level):
    """
    Processes a single .dat file to generate a load profile.

    Parameters:
    - dat_file_path (str): Path to the .dat load data file.
    - beam_length (float): Beam length in meters.
    - R (float): Normalization factor for denormalizing r/R.
    - save_dir (str): Directory to save load profiles.
    - logging_dir (str): Directory to save log files.
    - log_level (str): Logging verbosity level.

    Returns:
    - None
    """
    try:
        # Extract TSRX name
        tsrx_name = extract_tsrx_name(dat_file_path)
    except ValueError as ve:
        print(f"Error: {ve}")
        return
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup logging for this file
    logger = setup_logging(log_level, logging_dir, tsrx_name, timestamp)
    
    logger.info(f"Processing file: '{dat_file_path}'")
    
    try:
        # Read load data
        load_df_normalized = read_load_data(dat_file_path, logger)
        
        # Denormalize loads
        load_df_denormalized = denormalize_loads(load_df_normalized, R, logger)
        
        # Shift load profile
        load_df_shifted = shift_load_profile(load_df_denormalized, logger)
        
        # Validate that x does not exceed beam length after shifting
        max_x = load_df_shifted['x'].max()
        if max_x > beam_length:
            logger.error(f"Load position exceeds beam length after denormalization and shifting. Max x: {max_x} m, Beam length: {beam_length} m.")
            raise ValueError("Load position exceeds beam length after denormalization and shifting.")
        else:
            logger.info(f"All x positions are within the beam length of {beam_length} meters.")
        
        # Map load components
        load_df_mapped = map_load_components(load_df_shifted, logger)
        
        # Generate and save load profile
        generate_load_profile(load_df_mapped, save_dir, tsrx_name, timestamp, logger)
        
    except Exception as e:
        logger.error(f"Failed to process file '{dat_file_path}': {e}")
        # Continue with next file
        return

def main():
    """
    Main function to generate and save load profile files for all .dat files in the specified directory.
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Generate Load Profiles from Multiple Load Data Files.')
    parser.add_argument('--data_dir', type=str, default='pre_processing/load_library/data/tidal_benchmark',
                        help='Directory containing .dat load data files. Default: pre_processing/load_library/data/tidal_benchmark')
    parser.add_argument('--beam_length', type=float, default=8.0,
                        help='Beam length (L) in meters. Default: 8.0')
    parser.add_argument('--R', type=float, default=0.8,
                        help='Normalization factor R for denormalizing r/R. Default: 0.8')
    parser.add_argument('--save_dir', type=str, default='pre_processing/load_library/load_profiles/tidal_benchmark',
                        help='Directory to save load profiles. Default: pre_processing/load_library/load_profiles/tidal_benchmark')
    parser.add_argument('--logging_dir', type=str, default='pre_processing/load_library/schemes/logging',
                        help='Directory to save log files. Default: pre_processing/load_library/schemes/logging')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level. Default: INFO')
    
    args = parser.parse_args()
    
    # Validate data_dir
    if not os.path.isdir(args.data_dir):
        print(f"Error: The specified data directory does not exist: '{args.data_dir}'")
        sys.exit(1)
    
    # Ensure save_dir exists
    try:
        os.makedirs(args.save_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create save directory '{args.save_dir}': {e}")
        sys.exit(1)
    
    # Ensure logging_dir exists
    try:
        os.makedirs(args.logging_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create logging directory '{args.logging_dir}': {e}")
        sys.exit(1)
    
    # Find all .dat files in data_dir
    dat_files = [os.path.join(args.data_dir, file) for file in os.listdir(args.data_dir) if file.lower().endswith('.dat')]
    
    if not dat_files:
        print(f"No .dat files found in directory '{args.data_dir}'. Exiting.")
        sys.exit(0)
    
    print(f"Found {len(dat_files)} .dat files in '{args.data_dir}'. Starting processing...")
    
    # Process each .dat file
    for dat_file in dat_files:
        process_dat_file(dat_file, args.beam_length, args.R, args.save_dir, args.logging_dir, args.log_level)
    
    print("All files processed. Check log files for details.")

if __name__ == "__main__":
    main()
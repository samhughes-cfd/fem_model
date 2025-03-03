import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import os
import glob

def find_stiffness_files(job_dir):
    """
    Finds and returns the paths of stiffness matrix files inside primary_results/ and results/<job_name>/primary_results/.
    Ignores job prefix and timestamps in filenames.
    """
    primary_results_dir = os.path.join(job_dir, "primary_results")
    results_dir = os.path.join(job_dir, "results", os.path.basename(job_dir), "primary_results")

    # Define expected filename patterns (ONLY based on key part of filenames, ignoring timestamps)
    filename_patterns = {
        "element_K": "static_element_K_e_*.txt",
        "global_K": "static_global_K_global_*.txt",
        "global_K_mod": "static_global_K_mod_*.txt",
        "global_K_cond": "static_global_K_cond_*.txt"
    }

    # Resolve actual file paths
    resolved_files = {}

    print("\n🔍 Searching for stiffness matrix files (ignoring timestamps & job prefix)...")

    for key, pattern in filename_patterns.items():
        # Search in primary_results first
        primary_pattern = os.path.join(primary_results_dir, f"*{pattern}")
        matched_files = sorted(glob.glob(primary_pattern))  # Use wildcard to match any timestamp

        # If looking for global_K_mod, search in results_dir instead
        if not matched_files and key == "global_K_mod":
            results_pattern = os.path.join(results_dir, f"*{pattern}")
            matched_files = sorted(glob.glob(results_pattern))

        if matched_files:
            resolved_files[key] = matched_files[-1]  # Pick the most recent file (last in sorted list)
            print(f"✔ Found: {resolved_files[key]}")
        else:
            print(f"❌ No matching files found for: {pattern}")

    return resolved_files

def parse_stiffness_data(file_path):
    """
    Parses a stiffness matrix data file and returns it as a NumPy array.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    matrix_entries = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):  # Ignore comments
            parts = line.split(',')
            row, col, value = int(parts[0].strip()), int(parts[1].strip()), float(parts[2].strip())
            matrix_entries.append((row, col, value))

    return np.array(matrix_entries)

import os
import numpy as np

import os
import numpy as np

def write_stiffness_matrices(stiffness_matrices, output_dir):
    """
    Writes stiffness matrices to structured .txt files with fully aligned column headers.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for key, matrix_data in stiffness_matrices.items():
        output_file = os.path.join(output_dir, f"{key}.txt")

        # Get unique global indices and create a mapping
        unique_indices = sorted(set(matrix_data[:, 0]) | set(matrix_data[:, 1]))
        index_map = {val: i for i, val in enumerate(unique_indices)}

        size = len(unique_indices)  # Define size based on number of unique indices
        formatted_matrix = np.zeros((size, size))

        # Populate full stiffness matrix
        for row, col, value in matrix_data:
            formatted_matrix[index_map[int(row)], index_map[int(col)]] = value

        # Determine column width dynamically based on max index length and data width
        col_width = max(len(str(int(idx))) for idx in unique_indices) + 2  # Add padding
        data_width = 12  # Fixed width for data values
        full_width = max(col_width, data_width)  # Ensure alignment

        # Write matrix with labeled global indices
        with open(output_file, 'w') as f:
            f.write(f"Stiffness Matrix: {key}\n\n")

            # Header row (aligned with data entries)
            f.write(" " * col_width)  # Align first header cell
            f.write("".join(f"{int(idx):>{full_width}}" for idx in unique_indices) + "\n")

            # Data rows
            for idx in unique_indices:
                row_values = "".join(f"{formatted_matrix[index_map[int(idx)], index_map[int(j)]]:{full_width}.2e}"
                                     for j in unique_indices)
                f.write(f"{int(idx):>{col_width}}" + row_values + "\n")

    print(f"✅ Stiffness matrices saved in: {output_dir}")

def plot_stiffness_matrix(matrix_data, title, output_dir):
    """
    Generates and saves a heatmap visualization of the stiffness matrix.
    Zero values are set to white for better visibility.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    unique_indices = sorted(set(matrix_data[:, 0]) | set(matrix_data[:, 1]))
    index_map = {val: i for i, val in enumerate(unique_indices)}
    size = len(unique_indices)

    stiffness_matrix = np.zeros((size, size))

    for row, col, value in matrix_data:
        stiffness_matrix[index_map[int(row)], index_map[int(col)]] = value

    # Define colormap with white for zeros
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='white')  # Set background (NaN) to white
    masked_matrix = np.ma.masked_where(stiffness_matrix == 0, stiffness_matrix)  # Mask zeros

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(np.log1p(np.abs(masked_matrix)), cmap=cmap, aspect='auto', norm=mcolors.LogNorm())

    ax.set_xticks(range(len(unique_indices)))
    ax.set_xticklabels([int(idx) for idx in unique_indices], rotation=90)

    ax.set_yticks(range(len(unique_indices)))
    ax.set_yticklabels([int(idx) for idx in unique_indices])

    ax.set_xlabel("Column Index (Global)")
    ax.set_ylabel("Row Index (Global)")
    ax.set_title(f"Stiffness Matrix: {title}")

    fig.colorbar(cax, label='Log Magnitude')

    output_file = os.path.join(output_dir, f"{title}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Heatmap saved to {output_file}")

def plot_normalized_stiffness_matrix(matrix_data, title, output_dir):
    """
    Generates a normalized heatmap visualization of the stiffness matrix.
    Zero values are set to white.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    unique_indices = sorted(set(matrix_data[:, 0]) | set(matrix_data[:, 1]))
    index_map = {val: i for i, val in enumerate(unique_indices)}
    size = len(unique_indices)

    stiffness_matrix = np.zeros((size, size))

    for row, col, value in matrix_data:
        stiffness_matrix[index_map[int(row)], index_map[int(col)]] = value

    # **2D Normalization**: Normalize entire matrix based on max absolute value
    max_value = np.max(np.abs(stiffness_matrix))
    if max_value > 0:  
        normalized_matrix = stiffness_matrix / max_value
    else:
        normalized_matrix = stiffness_matrix  # Keep as-is if all zeros

    # Define colormap with white for zeros
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='white')  
    masked_matrix = np.ma.masked_where(normalized_matrix == 0, normalized_matrix)  # Mask zeros

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(masked_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(range(len(unique_indices)))
    ax.set_xticklabels([int(idx) for idx in unique_indices], rotation=90)

    ax.set_yticks(range(len(unique_indices)))
    ax.set_yticklabels([int(idx) for idx in unique_indices])

    ax.set_xlabel("Column Index (Global)")
    ax.set_ylabel("Row Index (Global)")
    ax.set_title(f"Normalized Stiffness Matrix: {title}")

    fig.colorbar(cax, label='Normalized Magnitude (-1 to 1)')

    output_file = os.path.join(output_dir, f"{title}_normalized.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Normalized heatmap saved to {output_file}")

def plot_stiffness_distribution(matrix_data, title, output_dir):
    """
    Plots a histogram of absolute stiffness values to show their distribution.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    values = np.abs(matrix_data[:, 2])  # Extract stiffness values

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, log=True, edgecolor='black')

    plt.xlabel("Absolute Stiffness Value")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Stiffness Values: {title}")

    output_file = os.path.join(output_dir, f"{title}_distribution.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Stiffness value distribution plot saved to {output_file}")

def load_and_process_stiffness_matrices(job_dir, output_base_dir):
    """
    Finds, parses, writes, and plots stiffness matrices.
    Saves all results in post_processing/tensor_visualisers/stiffness_matrix_evolution/
    """
    stiffness_files = find_stiffness_files(job_dir)
    stiffness_matrices = {}

    job_name = os.path.basename(job_dir)
    output_dir = os.path.join(output_base_dir, "stiffness_matrix_evolution", job_name)  
    # Store results in tensor_visualisers/stiffness_matrix_evolution/<job_name>/

    print(f"\n📝 Output directory: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("📁 Created output directory.")

    # Load stiffness matrices
    for key, file_path in stiffness_files.items():
        if os.path.exists(file_path):
            print(f"📖 Reading: {file_path}")
            stiffness_matrices[key] = parse_stiffness_data(file_path)
        else:
            print(f"⚠ Skipping missing file: {file_path}")

    if not stiffness_matrices:
        print("❌ No stiffness matrices found. Exiting...")
        return

    # Write structured matrices
    write_stiffness_matrices(stiffness_matrices, output_dir)

    # Plot stiffness matrices
    for key, matrix in stiffness_matrices.items():
        plot_stiffness_matrix(matrix, key, output_dir)
        plot_normalized_stiffness_matrix(matrix, key, output_dir)
        plot_stiffness_distribution(matrix, key, output_dir)  # Added histogram

    print(f"\n✅ Processing complete. Results saved in: {output_dir}")
    return stiffness_matrices

# Example Usage:
if __name__ == "__main__":
    base_directory = "post_processing/results"  # Parent directory where job folders are located
    output_directory = "post_processing/tensor_visualisers"  # New base directory for storing results
    selected_job = "job_0006_2025-03-03_09-41-09"  # Selected job folder

    job_directory = os.path.join(base_directory, selected_job)

    if os.path.exists(job_directory):
        load_and_process_stiffness_matrices(job_directory, output_directory)
    else:
        print(f"Error: Job directory {job_directory} does not exist.")
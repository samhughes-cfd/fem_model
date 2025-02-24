import os
import numpy as np
import matplotlib.pyplot as plt

def parse_stiffness_data(file_path, is_elemental=False):
    """
    Parses a stiffness matrix data file and organizes it into a dictionary.
    If is_elemental is True, the function will handle multiple element IDs separately.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data_matrices = {}
    current_key = None
    current_element = None
    current_matrix = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('# Data key:'):
            if current_key and current_matrix:
                data_matrices[current_key] = np.array(current_matrix)
            current_key = line.split(':')[-1].strip()
            current_matrix = []
        elif is_elemental and line.startswith('# Element ID:'):
            if current_element is not None and current_matrix:
                data_matrices[f"{current_key}_element_{current_element}"] = np.array(current_matrix)
            current_element = line.split(':')[-1].strip()
            current_matrix = []
        elif line and not line.startswith('#'):
            parts = line.split(',')
            row, col, value = int(parts[0].strip()), int(parts[1].strip()), float(parts[2].strip())
            current_matrix.append((row, col, value))
    
    if current_key and current_matrix:
        if is_elemental:
            data_matrices[f"{current_key}_element_{current_element}"] = np.array(current_matrix)
        else:
            data_matrices[current_key] = np.array(current_matrix)
    
    return data_matrices

def write_stiffness_matrices(data_matrices, output_dir):
    """
    Writes stiffness matrices into readable .txt files with labeled indices.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for key, matrix in data_matrices.items():
        output_file = os.path.join(output_dir, f"{key}.txt")
        unique_indices = sorted(set(matrix[:, 0]) | set(matrix[:, 1]))
        index_map = {val: i for i, val in enumerate(unique_indices)}
        
        max_idx = max(unique_indices) + 1
        formatted_matrix = np.zeros((max_idx, max_idx))
        for row, col, value in matrix:
            formatted_matrix[int(row), int(col)] = value
        
        with open(output_file, 'w') as f:
            f.write(f"Stiffness Matrix: {key}\n")
            f.write("\t" + "\t".join(map(str, unique_indices)) + "\n")
            for i, idx in enumerate(unique_indices):
                f.write(f"{idx}\t" + "\t".join(f"{formatted_matrix[idx, j]:.2e}" for j in unique_indices) + "\n")
    
    print(f"Stiffness matrices written to {output_dir}")

def plot_stiffness_matrix(matrix_data, title, output_dir):
    """
    Generates a heatmap visualization of the stiffness matrix.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    unique_indices = sorted(set(matrix_data[:, 0]) | set(matrix_data[:, 1]))
    size = max(unique_indices) + 1
    stiffness_matrix = np.zeros((size, size))
    
    for row, col, value in matrix_data:
        stiffness_matrix[int(row), int(col)] = value
    
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log1p(np.abs(stiffness_matrix)), cmap='viridis', aspect='auto')
    plt.colorbar(label='Log Magnitude')
    plt.title(f"Stiffness Matrix: {title}")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    
    output_file = os.path.join(output_dir, f"{title}.png")
    plt.savefig(output_file, dpi=300)
    plt.show()
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    input_file = "stiffness_data.txt"  # Replace with actual file path
    output_directory = "stiffness_matrices"
    
    # Parsing elemental and global stiffness matrices separately
    stiffness_data_elemental = parse_stiffness_data(input_file, is_elemental=True)
    stiffness_data_global = parse_stiffness_data(input_file, is_elemental=False)
    
    # Writing structured stiffness matrices to files
    write_stiffness_matrices(stiffness_data_elemental, output_directory)
    write_stiffness_matrices(stiffness_data_global, output_directory)
    
    # Plotting each matrix
    for key, matrix in {**stiffness_data_elemental, **stiffness_data_global}.items():
        plot_stiffness_matrix(matrix, key, output_directory)

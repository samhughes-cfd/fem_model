import numpy as np
import os

def read_stiffness_matrices(filename):
    elements = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if "Element ID:" in lines[i]:
                element_id = int(lines[i].split()[-1])
                i += 1
                stiffness_matrix = np.zeros((12, 12))
                while i < len(lines) and not lines[i].startswith("#"):
                    if lines[i].strip():
                        row, col, value = map(float, map(str.strip, lines[i].strip().split(',')))
                        stiffness_matrix[int(row), int(col)] = value
                    i += 1
                elements[element_id] = stiffness_matrix
            else:
                i += 1
    return elements

def read_force_vectors(filename):
    elements = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if "Element ID:" in lines[i]:
                element_id = int(lines[i].split()[-1])
                i += 1
                force_vector = np.array(list(map(float, lines[i].strip().split(','))))
                elements[element_id] = force_vector
                i += 1
            else:
                i += 1
    return elements

def save_matrix_to_file(matrix, filename, output_dir):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as file:
        file.write(f"# {filename}\n\n")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] != 0:  # Only non-zero values are saved
                    file.write(f"{i}, {j}, {matrix[i, j]:.6e}\n")
            file.write("\n")

def save_vector_to_file(vector, filename, output_dir):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as file:
        file.write(f"# {filename}\n\n")
        file.write(" ".join(f"{v:.6e}" for v in vector) + "\n")
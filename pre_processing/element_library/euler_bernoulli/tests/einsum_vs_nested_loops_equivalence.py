import numpy as np
import time
import matplotlib.pyplot as plt

# Function for stiffness calculation using einsum
def compute_stiffness_einsum(weights, B_T_tensor, D, B_tensor, detJ):
    return np.einsum("i,ijk,kl,ilm->jm", weights, B_T_tensor, D, B_tensor) * detJ

# Function for stiffness calculation using nested loops
def compute_stiffness_loops(weights, B_T_tensor, D, B_tensor, detJ):
    num_gauss_points = weights.shape[0]
    size = B_T_tensor.shape[1]  # Should be 6 for a 6x6 matrix
    Ke = np.zeros((size, size))  # Initialize stiffness matrix
    
    for i in range(num_gauss_points):
        for j in range(size):
            for m in range(size):
                sum_term = 0.0
                for k in range(2):  # Assuming D is 2x2
                    for l in range(2):
                        sum_term += B_T_tensor[i, j, k] * D[k, l] * B_tensor[i, l, m]
                Ke[j, m] += weights[i] * sum_term * detJ
    return Ke

# Generate random test data with correct shapes
num_gauss_points = 3
dof_size = 6  # 6x6 stiffness matrix
dim = 2  # Dimension of D matrix

weights = np.random.rand(num_gauss_points)  # Shape: (3,)
B_T_tensor = np.random.rand(num_gauss_points, dof_size, dim)  # Shape: (3,6,2)
D = np.random.rand(dim, dim)  # Shape: (2,2)
B_tensor = np.random.rand(num_gauss_points, dim, dof_size)  # Shape: (3,2,6)
detJ = np.random.rand()  # Random determinant value

# Run one pass of each method
Ke_einsum = compute_stiffness_einsum(weights, B_T_tensor, D, B_tensor, detJ)
Ke_loops = compute_stiffness_loops(weights, B_T_tensor, D, B_tensor, detJ)

# Print results
print("\n🔹 Einsum Computed Stiffness Matrix:\n", Ke_einsum)
print("\n🔹 Nested Loops Computed Stiffness Matrix:\n", Ke_loops)

# Compute and print the element-wise difference
difference = Ke_einsum - Ke_loops
print("\n🔸 Element-wise Difference:\n", difference)

# Check if matrices are nearly identical
if np.allclose(Ke_einsum, Ke_loops, atol=1e-10):
    print("\n✅ The stiffness matrices are equal within a tolerance of 1e-10.\n")
else:
    print("\n❌ The stiffness matrices are NOT equal! Check for precision issues.\n")
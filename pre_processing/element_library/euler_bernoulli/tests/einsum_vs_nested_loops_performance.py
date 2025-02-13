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
num_trials = 10_000  # Number of trials to measure performance

# Store times
einsum_times = []
loop_times = []

for _ in range(num_trials):
    weights = np.random.rand(num_gauss_points)  # Shape: (3,)
    B_T_tensor = np.random.rand(num_gauss_points, dof_size, dim)  # Shape: (3,6,2)
    D = np.random.rand(dim, dim)  # Shape: (2,2)
    B_tensor = np.random.rand(num_gauss_points, dim, dof_size)  # Shape: (3,2,6)
    detJ = np.random.rand()  # Random determinant value

    # Measure time for Einsum computation
    start_einsum = time.time()
    _ = compute_stiffness_einsum(weights, B_T_tensor, D, B_tensor, detJ)
    end_einsum = time.time()
    einsum_times.append(end_einsum - start_einsum)

    # Measure time for Nested Loops computation
    start_loops = time.time()
    _ = compute_stiffness_loops(weights, B_T_tensor, D, B_tensor, detJ)
    end_loops = time.time()
    loop_times.append(end_loops - start_loops)

# Compute averages
avg_einsum_time = np.mean(einsum_times)
avg_loop_time = np.mean(loop_times)

# Print results
print(f"\n⏱ Average Einsum Computation Time: {avg_einsum_time:.6f} seconds per iteration")
print(f"⏱ Average Nested Loops Computation Time: {avg_loop_time:.6f} seconds per iteration")

# Performance ratio
speedup = avg_loop_time / avg_einsum_time
print(f"\n🚀 Einsum is approximately {speedup:.2f}x faster than Nested Loops!\n")

# Normalize x-axis by maximum observed time
max_time = max(max(einsum_times), max(loop_times))
einsum_times_normalized = np.array(einsum_times) / max_time
loop_times_normalized = np.array(loop_times) / max_time

# Create histogram and CDF plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram plot (normalized)
axes[0].hist(einsum_times_normalized, bins=1000, alpha=0.7, label="Einsum Times", color="blue", density=True)
axes[0].hist(loop_times_normalized, bins=1000, alpha=0.7, label="Loop Times", color="red", density=True)
axes[0].set_xlabel("Normalized Computation Time")
axes[0].set_ylabel("Probability Density")
axes[0].set_title("Normalized Performance Distribution of Einsum vs Nested Loops")
axes[0].legend()

# Compute and plot CDF
einsum_sorted = np.sort(einsum_times_normalized)
loop_sorted = np.sort(loop_times_normalized)
cdf_einsum = np.arange(1, len(einsum_sorted) + 1) / len(einsum_sorted)
cdf_loop = np.arange(1, len(loop_sorted) + 1) / len(loop_sorted)

axes[1].plot(einsum_sorted, cdf_einsum, label="Einsum CDF", color="blue")
axes[1].plot(loop_sorted, cdf_loop, label="Loop CDF", color="red")
axes[1].set_xlabel("Normalized Computation Time")
axes[1].set_ylabel("Cumulative Probability")
axes[1].set_title("CDF of Normalized Computation Times")
axes[1].legend()

plt.tight_layout()
plt.show()
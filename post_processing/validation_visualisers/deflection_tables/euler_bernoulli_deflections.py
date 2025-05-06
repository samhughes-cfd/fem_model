# post_processing\validation_visualisers\deflection_tables\euler_bernoulli_deflections.py

import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Constants
E = 2e11  # Young's Modulus in Pa (N/m^2)
I_y = 6.6667e-5  # Second moment of area in m^4
F = 1e5  # Point Load in N
q_0 = 1e5  # Uniform Load Intensity in N/m

# Job files and corresponding load types
job_files = [
    "post_processing/results/job_0001_2025-05-06_15-47-53/primary_results/job_0001_static_global_U_global_2025-05-06_15-47-56.txt",
    "post_processing/results/job_0002_2025-05-06_15-47-53/primary_results/job_0002_static_global_U_global_2025-05-06_15-47-56.txt",
    "post_processing/results/job_0003_2025-05-06_15-47-53/primary_results/job_0003_static_global_U_global_2025-05-06_15-47-56.txt",
    "post_processing/results/job_0004_2025-05-06_15-47-53/primary_results/job_0004_static_global_U_global_2025-05-06_15-47-59.txt",
    "post_processing/results/job_0005_2025-05-06_15-47-53/primary_results/job_0005_static_global_U_global_2025-05-06_15-47-59.txt",
    "post_processing/results/job_0006_2025-05-06_15-47-53/primary_results/job_0006_static_global_U_global_2025-05-06_15-47-59.txt",
]

job_to_loadtype = {
    "job_0001": "End Load",
    "job_0002": "Midpoint Load",
    "job_0003": "Quarterpoint Load",
    "job_0004": "Constant Distributed Load",
    "job_0005": "Quadratic Distributed Load",
    "job_0006": "Parabolic Distributed Load"
}

# Mesh node positions (based on the provided mesh)
nodes = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])  # Node positions in meters

def get_nodewise_displacement(filepath):
    # Read the displacement data (6 DOF per node)
    data = np.loadtxt(filepath)
    # Extracting the vertical displacement (assuming it's the second value in the 6 DOF data)
    uy = data[1::6]  # Every 6th value starting from the second column
    return uy

def generate_expected_shape(x, load_type):
    L = x[-1]
    
    # Normalize x
    xi = x / L
    
    if load_type == "End Load":
        return (F * x**2) / (6 * E * I_y) * (3 * L - x)  # Point Load at Free End (x=L)
    elif load_type == "Midpoint Load":
        return (F * x**2) / (6 * E * I_y) * (3 * L - x)  # Simplified for midpoint load
    elif load_type == "Quarterpoint Load":
        return (F * x**2) / (6 * E * I_y) * (3 * L - x)  # Approximation for quarter point
    elif load_type == "Constant Distributed Load":
        return (q_0 * x**2) / (24 * E * I_y) * (6 * L**2 - 4 * L * x + x**2)  # Uniform Load
    elif load_type == "Quadratic Distributed Load":
        return (q_0 * x**4) / (30 * L**2) - (q_0 * x**5) / (10 * L) + (q_0 * x**6) / (12)  # Quadratic Load
    elif load_type == "Parabolic Distributed Load":
        return (q_0 * x**5) / (20 * L**2) - (q_0 * x**6) / (30 * L**3) - (q_0 * x**4) / (12)  # Parabolic Load
    else:
        return np.zeros_like(x)

# Function to extract job name from file path using regular expressions (regex)
def extract_job_name(file_path):
    match = re.search(r'job_\d+', os.path.basename(file_path))  # Match "job_<number>"
    if match:
        return match.group(0)  # Return the matched job name (e.g., 'job_0001')
    else:
        return "Unknown"  # Return "Unknown" if no match is found

# Define beam nodes (using the provided mesh)
n_nodes = len(nodes)
L = nodes[-1]  # beam length from the last node
x = nodes  # Beam node positions

# Create a 2x3 subplot grid
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# Flatten the 2x3 grid to iterate through each subplot
axs = axs.flatten()

for i, job_file in enumerate(job_files):
    # Extract job name from the file path and map it to load type
    job_name = extract_job_name(job_file)  # Use the robust job name extraction
    load_type = job_to_loadtype.get(job_name, "Unknown")  # Get the load type from the mapping

    try:
        uy = get_nodewise_displacement(job_file)  # Extract the vertical displacement (uy)
    except Exception as e:
        print(f"Could not load {job_file}: {e}")
        continue

    # Convert FEM results (uy) from meters to millimeters
    uy_mm = uy * 1000

    expected = generate_expected_shape(x, load_type)

    # Selectively apply negative sign to expected displacement, skipping Parabolic Load
    if load_type != "Parabolic Distributed Load":
        expected_mm = expected * 1000 * -1  # Make expected displacement negative
    else:
        expected_mm = expected * 1000  # Keep it positive for parabolic load

    # Plot FEM displacement and expected shape in the subplot grid
    ax = axs[i]
    ax.plot(x, uy_mm, label=f"{load_type} - FEM", linestyle='-', marker='o', markersize=8)
    ax.plot(x, expected_mm, label=f"{load_type} - Expected", linestyle='--', linewidth=2)
    ax.set_xlabel("Beam Length (x) [m]", fontsize=12)
    ax.set_ylabel("Vertical Displacement (uy) [mm]", fontsize=12)
    ax.set_title(f"{load_type} Comparison", fontsize=14)  # Set the title with the correct load type
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=10)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
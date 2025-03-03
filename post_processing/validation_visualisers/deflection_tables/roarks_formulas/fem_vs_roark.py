import os
import numpy as np
import matplotlib.pyplot as plt

# === HARDCODE SPECIFIC JOB FILES TO READ ===
job_files = [
    "post_processing/results/job_0001_2025-03-03_09-41-09/primary_results/job_0001_static_global_U_global_2025-03-03_09-41-09.txt",
    "post_processing/results/job_0002_2025-03-03_09-41-09/primary_results/job_0002_static_global_U_global_2025-03-03_09-41-09.txt",
    "post_processing/results/job_0003_2025-03-03_09-41-09/primary_results/job_0003_static_global_U_global_2025-03-03_09-41-09.txt",
]

# Beam parameters
L = 8.0           # [m]
E = 2.0e11        # [Pa]
I = 2.67e-7       # [m^4]

def extract_and_compute_quantities(job_files):
    results = {}

    for file_path in job_files:
        job_name = os.path.basename(file_path).split("_static")[0]  # Extract job name
        
        print(f"Processing: {file_path}")

        try:
            with open(file_path, "r") as file:
                lines = file.readlines()

                # Remove comment/header lines
                data_lines = [line.strip() for line in lines if not line.startswith("#")]

                # Convert to float array
                U_global = np.array([float(value) for value in data_lines])

                # Number of nodes
                num_nodes = len(U_global) // 6  # Since each node has 6 DOFs
                x_vals = np.linspace(0, L, num_nodes)

                print(f"File {job_name}: Read {num_nodes} nodes.")

                # Extract u_y values (skip the first fixed node)
                uy_values = np.array([U_global[6*i + 1] for i in range(1, num_nodes)])

                if len(uy_values) == 0:
                    print(f"Warning: No u_y values extracted from {job_name}!")

                # Compute θ_z (rotation)
                theta_z_values = np.zeros(len(uy_values))
                for i in range(len(uy_values)):
                    if i == 0:
                        theta_z_values[i] = (-3 * uy_values[i] + 4 * uy_values[i + 1] - uy_values[i + 2]) / (2 * L)
                    elif i == len(uy_values) - 1:
                        theta_z_values[i] = (3 * uy_values[i] - 4 * uy_values[i - 1] + uy_values[i - 2]) / (2 * L)
                    else:
                        theta_z_values[i] = (uy_values[i + 1] - uy_values[i]) / L

                # Compute M(x) (bending moment)
                Mx_values = np.zeros(len(uy_values))
                for i in range(len(uy_values)):
                    if i == 0:
                        Mx_values[i] = (2 * uy_values[i] - 5 * uy_values[i + 1] + 4 * uy_values[i + 2] - uy_values[i + 3]) * (E * I / L**2)
                    elif i == len(uy_values) - 1:
                        Mx_values[i] = (2 * uy_values[i] - 5 * uy_values[i - 1] + 4 * uy_values[i - 2] - uy_values[i - 3]) * (E * I / L**2)
                    else:
                        Mx_values[i] = (uy_values[i + 1] - 2 * uy_values[i] + uy_values[i - 1]) * (E * I / L**2)

                # Compute V(x) (shear force)
                Vx_values = np.zeros(len(uy_values))
                for i in range(1, len(uy_values) - 1):
                    Vx_values[i] = (Mx_values[i + 1] - Mx_values[i - 1]) / (2 * L)

                # Store results
                results[job_name] = {
                    "x": x_vals[1:],  # Exclude fixed node
                    "u_y": uy_values,
                    "theta_z": theta_z_values,
                    "V_x": Vx_values,
                    "M_x": Mx_values
                }

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return results

# Run processing
results = extract_and_compute_quantities(job_files)

# === PLOTTING ===
fig, axes = plt.subplots(nrows=4, ncols=len(job_files), figsize=(15, 12), sharex=True)

for col_idx, (job_name, data) in enumerate(results.items()):
    x = data["x"]
    
    # Plot u_y
    axes[0, col_idx].plot(x, data["u_y"], color="blue")
    axes[0, col_idx].set_ylabel(r"$u_{y}(x)\,[mm]$")
    axes[0, col_idx].set_title(f"{job_name}")

    # Plot θ_z
    axes[1, col_idx].plot(x, data["theta_z"], color="orange")
    axes[1, col_idx].set_ylabel(r"$\theta_{z}(x)\,[^\circ]$")

    # Plot V(x)
    axes[2, col_idx].plot(x, data["V_x"], color="green")
    axes[2, col_idx].set_ylabel(r"$V(x)\,[kN]$")

    # Plot M(x)
    axes[3, col_idx].plot(x, data["M_x"], color="red")
    axes[3, col_idx].set_ylabel(r"$M(x)\,[kN \cdot m]$")
    axes[3, col_idx].set_xlabel(r"$x\,[m]$")

plt.tight_layout()
plt.show()

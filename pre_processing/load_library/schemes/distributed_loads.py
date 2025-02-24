import numpy as np
import os

# Beam parameters
L = 8.0         # Beam length (m)
w_max = 1000    # Maximum distributed load (N/m)
num_points = 100  # Number of discretized points

# Generate x values along the beam
x_vals = np.linspace(0, L, num_points)

# Compute distributed loads w(x)
w_udl = np.full_like(x_vals, w_max)        # UDL: Constant
w_triangular = (w_max / L) * x_vals        # Triangular: Linearly increasing
w_parabolic = (w_max / L**2) * x_vals**2   # Parabolic: Quadratically increasing

# Define the output directory
output_dir = "pre_processing/load_library/load_profiles/deflection_tables/distributed_loads"
os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

# Function to save load data in the correct format
def save_load_file(filename, x_vals, w_x):
    """Saves the distributed load data in the required format."""
    with open(filename, "w") as f:
        f.write("[Loads]\n")
        f.write("[x]         [y]         [z]         [F_x]         [F_y]         [F_z]         [M_x]         [M_y]         [M_z]\n")

        for i in range(len(x_vals)):
            # Force components (load applied in -y direction)
            F_x, F_y, F_z = 0.0, -w_x[i], 0.0

            # Moment components (bending about z-axis)
            M_x, M_y, M_z = 0.0, 0.0, -w_x[i] * x_vals[i]

            # Write the formatted line
            f.write(f"{x_vals[i]:.6f}    0.000000    0.000000    {F_x:.6f}    {F_y:.6f}    {F_z:.6f}    {M_x:.6f}    {M_y:.6f}    {M_z:.6f}\n")

# Save distributed loads to respective files
save_load_file(os.path.join(output_dir, "load_udl.txt"), x_vals, w_udl)
save_load_file(os.path.join(output_dir, "load_triangular.txt"), x_vals, w_triangular)
save_load_file(os.path.join(output_dir, "load_parabolic.txt"), x_vals, w_parabolic)

print("Load files successfully generated in:", output_dir)
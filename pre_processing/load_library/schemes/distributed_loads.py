import numpy as np
import os

# Beam parameters
L = 8.0         # Beam length (m)
w_max = 1000    # Maximum distributed load (N/m)
num_points = 30  # Number of discretized points

# Generate x values along the beam
x_vals = np.linspace(0, L, num_points)

def roark_load_intensity(x, L, w, load_type):
    """
    Returns q(x) for the chosen load type (UDL, triangular, parabolic),
    with maximum at x = L (right-hand side).
    """
    if load_type == "udl":
        # Uniformly Distributed Load (UDL) remains constant
        return w * np.ones_like(x), "q(x) = w"
    elif load_type == "triangular":
        # Linearly increasing load: q(x) = w_max * (x / L)
        return w * (x / L), "q(x) = w * (x/L)"
    elif load_type == "parabolic":
        # Quadratically increasing load: q(x) = w_max * (x / L)^2
        return w * (x / L)**2, "q(x) = w * (x/L)^2"
    else:
        raise ValueError("Invalid load_type. Must be 'udl', 'triangular', or 'parabolic'.")

# Load profiles dictionary
load_profiles = {
    "load_udl": "udl",
    "load_triangular": "triangular",
    "load_parabolic": "parabolic"
}

# Define the output directory
output_dir = "pre_processing/load_library/load_profiles/deflection_tables/distributed_loads"
os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

# Function to format values and remove negative zero
def clean_float(value):
    """Formats float values and prevents negative zero."""
    return f"{value:.6f}".replace("-0.000000", "0.000000")

# Function to save load data in the required format
def save_load_file(filename, x_vals, w_x, load_type, formula):
    """Saves the distributed load data in the required format with a labeled header."""
    with open(filename, "w") as f:
        # Write file header with load type and formula
        f.write(f"# Load Type: {load_type.upper()}\n")
        f.write(f"# Load Formula: {formula}\n")
        f.write("[Loads]\n")
        f.write(f"{'[x]':>12} {'[y]':>12} {'[z]':>12} {'[F_x]':>12} {'[F_y]':>12} {'[F_z]':>12} {'[M_x]':>12} {'[M_y]':>12} {'[M_z]':>12}\n")

        for i in range(len(x_vals)):
            # Distributed force components
            F_x, F_y, F_z = 0.0, -w_x[i], 0.0  # Only F_y is nonzero

            # Moment components (set to zero, FEM solver computes internal moments)
            M_x, M_y, M_z = 0.0, 0.0, 0.0

            # Write the formatted line with proper spacing
            f.write(f"{clean_float(x_vals[i]):>12} {'0.000000':>12} {'0.000000':>12} {clean_float(F_x):>12} {clean_float(F_y):>12} {clean_float(F_z):>12} {clean_float(M_x):>12} {clean_float(M_y):>12} {clean_float(M_z):>12}\n")

# Save distributed loads to respective files
for filename, load_type in load_profiles.items():
    w_x, formula = roark_load_intensity(x_vals, L, w_max, load_type)
    save_load_file(os.path.join(output_dir, f"{filename}.txt"), x_vals, w_x, load_type, formula)

print("Load files successfully generated in:", output_dir)
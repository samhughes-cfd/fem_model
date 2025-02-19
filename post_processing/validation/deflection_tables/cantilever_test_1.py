# post_processing\validation\deflection_tables\cantilever_test_1.py

"""

Free End Point Load -
u_y = -(F * x**2 / (6 * E * I))*(3*L - x)
theta_z = -(F * x / (2 * E * I))*(2*L - x)

"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

mpl.rc('text', usetex=False)

# File paths
relative_path = os.path.join("results", "job_0001", "displacements_EulerBernoulliBeamElement_20241209_004137.txt")
plots_dir = "validation_plots"

# Create plots directory if it doesn't exist
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

if not os.path.exists(relative_path):
    raise FileNotFoundError(f"File not found: {relative_path}")

# Beam parameters
F = 1000.0  # Load [N]
E = 210e9   # Young's modulus [Pa]
I = 3.33e-5 # Moment of inertia [m^4] (Updated for C200x20 section)
L = 8.0     # Beam length [m]

# Theoretical functions
def w_theoretical(x, F, E, I, L):
    return -(F * x**2 / (6 * E * I))*(3*L - x)

def theta_theoretical(x, F, E, I, L):
    return -(F * x / (2 * E * I))*(2*L - x)

# Initialize lists for data
x_vals, w_vals, theta_vals = [], [], []

# Read the file
with open(relative_path, "r") as f:
    lines = f.readlines()

if len(lines) < 3:
    raise ValueError(f"Not enough lines in file {relative_path}")

# Parse the data
for line_num, line in enumerate(lines[2:], start=3):  # Skip first two lines (headers)
    parts = line.split()
    if len(parts) == 6 and parts[0] == "Node":
        try:
            position = float(parts[2])
            w = float(parts[4])
            theta = float(parts[5])
            x_vals.append(position)
            w_vals.append(w)
            theta_vals.append(theta)
        except ValueError:
            warnings.warn(f"Skipping malformed data at line {line_num}: {line.strip()}")

if not x_vals:
    raise ValueError(f"No valid data found in {relative_path}")

# Convert lists to numpy arrays
x_vals = np.array(x_vals)
w_vals = np.array(w_vals)
theta_vals = np.array(theta_vals)

# Compute analytical solutions
w_analytical = w_theoretical(x_vals, F, E, I, L)
theta_analytical = theta_theoretical(x_vals, F, E, I, L)

# Plot w(x)
plt.figure(figsize=(10, 6))
plt.plot(x_vals, w_vals, 'o', label='$w$', markersize=4)
plt.plot(x_vals, w_analytical, '-', label=r'$w(x) = -\frac{F x^{2}}{6 E I}(3L - x)$')
plt.title('Beam Deflection: FEM vs Analytical')
plt.xlabel('$x \, [\mathrm{m}]$')
plt.ylabel('$w \, [\mathrm{m}]$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "beam_deflection.png"))

# Plot θ(x)
plt.figure(figsize=(10, 6))
plt.plot(x_vals, theta_vals, 'o', label='$\\theta$', markersize=4)
plt.plot(x_vals, theta_analytical, '-', label=r'$\theta(x) = -\frac{F x}{2 E I}(2L - x)$')
plt.title('Beam Rotation: FEM vs Analytical')
plt.xlabel('$x \, [\mathrm{m}]$')
plt.ylabel('$\\theta \, [\mathrm{rad}]$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "beam_rotation.png"))

plt.show()
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

# Define job folders for each load type
job_folders = {
    "udl": ["job_0001", "job_0002", "job_0003"],
    "triangular": ["job_0004", "job_0005", "job_0006"],
    "parabolic": ["job_0007", "job_0008", "job_0009"]
}

# Global parameters
L = 8.0
w_max = 1000

# Color scheme for FEM results (same as used for analytical plots)
colors = {
    "deflection": "#4F81BD",  # Blue
    "rotation": "#4F81BD",    # Blue
    "shear": "#9BBB59",       # Green
    "moment": "#C0504D"       # Red
}

# --- Functions to read data ---

def read_mesh_file(mesh_file):
    """Reads nodal x-coordinates from a mesh file, skipping headers.
       Assumes that the first numeric line starts with a digit.
       The x-coordinate is in the second column.
    """
    with open(mesh_file, 'r') as file:
        lines = file.readlines()

    # Find first line that starts with a digit
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and stripped[0].isdigit():
            data_start = i
            break

    # Load data from that line onward, extracting column 1 (index 1)
    mesh_data = np.loadtxt(lines[data_start:], usecols=[1])
    return mesh_data

def extract_fem_dofs(filename):
    """Extracts U_y, θ_z, F_y, M_z from a FEM results file.
       Skips header lines (starting with '#') and then:
         - U_y: every 6th value starting from index 1 (converted from m to mm)
         - θ_z: every 6th value starting from index 5 (converted from rad to °)
         - F_y: same indices as U_y (converted from N to kN)
         - M_z: same indices as θ_z (converted from Nm to kNm)
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Extract numeric data (ignore comment lines)
    data = np.array([float(line.strip()) for line in lines if not line.startswith("#")])
    if len(data) % 6 != 0:
        raise ValueError(f"Error: The FEM results file {filename} does not contain a multiple of 6 entries.")
    
    U_y = data[1::6] * 1000         # m -> mm
    theta_z = np.degrees(data[5::6])  # rad -> degrees
    F_y = data[1::6] / 1000          # N -> kN
    M_z = data[5::6] / 1000          # Nm -> kNm
    return U_y, theta_z, F_y, M_z

# --- Function to print FEM results in tabular form ---

def print_fem_results(x_coords, U_y, theta_z, F_y, M_z, load_type, mesh_id):
    """Prints FEM results in tabular form for a given load type and mesh case."""
    df = pd.DataFrame({
        "x (m)": x_coords,
        "u_z (mm)": U_y,
        "θ_z (°)": theta_z,
        "F_y (kN)": F_y,
        "M_z (kNm)": M_z
    })
    
    # Set up formatters for scientific notation (4 decimal places)
    fmt = {
        "x (m)": lambda x: f"{x:.4e}",
        "u_z (mm)": lambda x: f"{x:.4e}",
        "θ_z (°)": lambda x: f"{x:.4e}",
        "F_y (kN)": lambda x: f"{x:.4e}",
        "M_z (kNm)": lambda x: f"{x:.4e}"
    }
    
    print("\n" + "="*50)
    print(f"FEM Results for {load_type.capitalize()} Load - Mesh {mesh_id}")
    print("="*50)
    print(df.to_string(index=False, formatters=fmt))
    print("="*50 + "\n")

# --- Read FEM data and print tables ---

fem_results = {}

for lt, jobs in job_folders.items():
    fem_results[lt] = {}
    print(f"\n================ {lt.upper()} Load =================")
    # Print the distributed load formula and w_max for each load case
    if lt == "udl":
        print(f"Distributed load formula: w_udl = np.full_like(x, {w_max:.4e})")
    elif lt == "triangular":
        print(f"Distributed load formula: w_triangular = ({w_max:.4e} / {L:.4e}) * x")
    elif lt == "parabolic":
        print(f"Distributed load formula: w_parabolic = ({w_max:.4e} / ({L:.4e}**2)) * x**2")
    
    for job in jobs:
        mesh_file = f"jobs/{job}/mesh.txt"
        U_files = glob.glob(f"post_processing/results/{job}_*/primary_results/{job}_static_global_U_global_*.txt")
        F_files = glob.glob(f"post_processing/results/{job}_*/primary_results/{job}_static_global_F_mod_*.txt")
        
        if not U_files or not F_files:
            print(f"Warning: Missing FEM result files for {job}. Skipping...")
            continue
        
        U_file, F_file = U_files[0], F_files[0]
        
        x_coords = read_mesh_file(mesh_file)
        U_y, theta_z, F_y, M_z = extract_fem_dofs(U_file)
        num_nodes = len(x_coords)
        if len(U_y) != num_nodes:
            raise ValueError(f"Error: Mesh and FEM results mismatch for {job}")
        
        fem_results[lt][job] = {
            "x": x_coords,
            "U_y": U_y,
            "theta_z": theta_z,
            "F_y": F_y,
            "M_z": M_z
        }
        
        # Print table for this mesh case in scientific notation
        print_fem_results(x_coords, U_y, theta_z, F_y, M_z, lt, job)

# --- Plotting FEM data only ---

fig, axes = plt.subplots(4, 3, figsize=(15, 12))
fem_markers = ["o", "s", "d"]  # Different markers for different meshes

# Iterate over load types (columns) and plot each category (rows)
for i, lt in enumerate(["udl", "triangular", "parabolic"]):
    jobs = sorted(fem_results[lt].keys())
    for row, category in enumerate(["deflection", "rotation", "shear", "moment"]):
        for mesh_idx, job in enumerate(jobs):
            data_dict = fem_results[lt][job]
            x_coords = data_dict["x"]
            if category == "deflection":
                fem_y = data_dict["U_y"]
            elif category == "rotation":
                fem_y = data_dict["theta_z"]
            elif category == "shear":
                fem_y = data_dict["F_y"]
            elif category == "moment":
                fem_y = data_dict["M_z"]

            axes[row, i].scatter(
                x_coords, fem_y, facecolors='none', edgecolors=colors[category],
                marker=fem_markers[mesh_idx], label=f"FEM Mesh {mesh_idx+1}"
            )

        if row == 0:
            axes[row, i].set_title(f"{lt.capitalize()} Load")
        y_labels = ["Deflection (mm)", "Rotation (°)", "Shear Force (kN)", "Bending Moment (kNm)"]
        axes[row, i].set_ylabel(y_labels[row])
        if row == 3:
            axes[row, i].set_xlabel("Position x (m)")
        axes[row, i].legend(frameon=False)
        axes[row, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

plt.tight_layout()
plt.show()
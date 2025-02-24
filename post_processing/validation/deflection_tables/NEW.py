import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

# Define job folders for each load type
job_folders = {
    "udl": ["job_0001", "job_0002", "job_0003"],
    "triangular": ["job_0004", "job_0005", "job_0006"],
    "parabolic": ["job_0007", "job_0008", "job_0009"]
}

# Global parameters
L = 8.0
w_max = 1000

# Color scheme for FEM results
colors = {
    "deflection": "#4F81BD",  # Blue
    "rotation_fem": "#C0504D",  # Red (theta_z from FEM force vector)
    "rotation_derived": "#9BBB59",  # Green (theta_z from u_y curvature)
    "shear": "#9BBB59",       # Green
    "moment": "#C0504D"       # Red
}

# --- Functions to read data ---

def read_mesh_file(mesh_file):
    """Reads nodal x-coordinates from a mesh file."""
    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Mesh file {mesh_file} not found.")

    with open(mesh_file, 'r') as file:
        lines = file.readlines()

    # Identify first numeric line
    data_start = next((i for i, line in enumerate(lines) if line.strip() and line.strip()[0].isdigit()), None)
    if data_start is None:
        raise ValueError(f"Mesh file {mesh_file} does not contain valid numeric data.")

    return np.loadtxt(lines[data_start:], usecols=[1])

def extract_fem_dofs(filename):
    """Extracts U_y, θ_z from a FEM results file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"FEM result file {filename} not found.")

    with open(filename, 'r') as file:
        lines = [line.strip() for line in file.readlines() if not line.startswith("#")]

    try:
        data = np.array([float(line) for line in lines])
    except ValueError:
        raise ValueError(f"Invalid numeric values in FEM results file {filename}")

    if len(data) % 6 != 0:
        raise ValueError(f"Error: FEM results file {filename} does not contain a multiple of 6 entries.")

    return (
        data[1::6] * 1000,    # U_y: Convert m → mm
        np.degrees(data[5::6])  # θ_z: Convert rad → degrees
    )

def compute_derived_quantities(U_global, element_lengths, material_props, geometry_props):
    """
    Computes post-processed quantities: θ_z (nodal rotations), M(x) (internal bending moment),
    and V(x) (shear force) for Euler-Bernoulli beam elements.
    """
    logger.info("Computing derived EB beam quantities: θ_z, M(x), and V(x)...")
    
    E = material_props["E"]
    I_z = geometry_props["I_z"]
    EI = E * I_z
    num_nodes = len(U_global)
    
    theta_z_values = np.zeros(num_nodes)
    M_values = np.zeros(num_nodes)
    V_values = np.zeros(num_nodes)
    
    # Compute θ_z using finite differences
    for i in range(1, num_nodes - 1):
        L = element_lengths[i - 1] if i < len(element_lengths) else element_lengths[-1]
        theta_z_values[i] = (U_global[i + 1] - U_global[i - 1]) / (2 * L)
    
    # Compute M(x) using second derivative of U_y
    for i in range(1, num_nodes - 1):
        L = element_lengths[i - 1] if i < len(element_lengths) else element_lengths[-1]
        M_values[i] = EI * (U_global[i + 1] - 2 * U_global[i] + U_global[i - 1]) / (L ** 2)
    
    # Compute V(x) using first derivative of M(x)
    for i in range(1, num_nodes - 1):
        L = element_lengths[i - 1] if i < len(element_lengths) else element_lengths[-1]
        V_values[i] = (M_values[i + 1] - M_values[i - 1]) / (2 * L)
    
    return theta_z_values, M_values, V_values

# --- Read FEM data and compute derived values ---

fem_results = {}

for lt, jobs in job_folders.items():
    fem_results[lt] = {}

    for job in jobs:
        mesh_file = f"jobs/{job}/mesh.txt"
        U_files = glob.glob(f"post_processing/results/{job}_*/primary_results/{job}_static_global_U_global_*.txt")
        
        if not U_files:
            print(f"Warning: Missing FEM result files for {job}. Skipping...")
            continue

        try:
            x_coords = read_mesh_file(mesh_file)
            U_y, theta_z_fem = extract_fem_dofs(U_files[0])
        except Exception as e:
            print(f"Error processing {job}: {e}")
            continue
        
        element_lengths = np.diff(x_coords)
        material_props = {"E": 2e11}  # Example Young's modulus
        geometry_props = {"I_z": 1e-4}  # Example moment of inertia

        theta_z_derived, M_x, V_x = compute_derived_quantities(U_y, element_lengths, material_props, geometry_props)
        
        fem_results[lt][job] = {
            "x": x_coords,
            "U_y": U_y,
            "theta_z_fem": theta_z_fem,
            "theta_z_derived": theta_z_derived,
            "V_x": V_x,
            "M_x": M_x
        }

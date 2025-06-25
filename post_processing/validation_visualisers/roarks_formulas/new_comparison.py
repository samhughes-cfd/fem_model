import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# -------------------------------------------------------------------------
#  ROARK IMPORTS
# -------------------------------------------------------------------------
from roarks_formulas_point import roark_point_load_response
from roarks_formulas_distributed import roark_distributed_load_response

# -------------------------------------------------------------------------
#  DIRECTORIES & CONSTANTS
# -------------------------------------------------------------------------
SAVE_DIR = "post_processing/validation_visualisers/roarks_formulas/new"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------------------------------------------------
#  COLOR SCHEME
# -------------------------------------------------------------------------
COLORS = {
    "deflection": "#4F81BD",  # Blue
    "rotation":   "#4F81BD",  # Blue
    "shear":      "#9BBB59",  # Green
    "moment":     "#C0504D",  # Red
    "fea":        "black",    # FEA data in black
}

plot_info = {
    "deflection": {
        "unit_factor": 1000.0,             # [m] → [mm]
        "label_name":  r"$u_{y}(x)\,[mm]$"
    },
    "rotation": {
        "unit_factor": (180.0 / np.pi),    # [rad] → [deg]
        "label_name":  r"$\theta_{z}(x)\,[^\circ]$"
    },
    "shear": {
        "unit_factor": 1.0 / 1000.0,       # [N] → [kN]
        "label_name":  r"$V(x)\,[kN]$"
    },
    "moment": {
        "unit_factor": 1.0 / 1000.0,       # [N·m] → [kN·m]
        "label_name":  r"$M(x)\,[kN \cdot m]$"
    },
}

def convert_data(category, data_array):
    """Applies a scale factor from `plot_info` to the data array."""
    factor = plot_info[category]["unit_factor"]
    return factor * data_array

# -------------------------------------------------------------------------
#   MESH PARSER
# -------------------------------------------------------------------------
def parse_mesh(mesh_file_path):
    """
    Parses the mesh file to extract node positions
    Returns: dict {node_id: (x, y, z)}
    """
    node_coords = {}
    with open(mesh_file_path, 'r') as f:
        # Skip header lines
        for _ in range(2): next(f)
        
        for line in f:
            if line.strip() == "":
                continue
            parts = line.split()
            try:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                node_coords[node_id] = (x, y, z)
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping malformed line: {line.strip()} - {str(e)}")
                continue
    return node_coords

# -------------------------------------------------------------------------
#   EXTRACT & COMPUTE QUANTITIES
# -------------------------------------------------------------------------
def extract_and_compute_quantities(job_files, mesh_coords, E, I, job_to_loadtype):
    """
    Reads each job-file and extracts:
       u_y, theta_z, V, M
    from the nodal displacements using actual node positions.
    """
    results = {}
    beam_length = max(x for x, _, _ in mesh_coords.values())
    
    for file_path in job_files:
        try:
            # Extract job name from directory path
            dir_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            job_name = None
            for job_id in job_to_loadtype:
                if job_id in dir_name:
                    job_name = job_id
                    break
            
            if not job_name:
                print(f"Job name not found for {dir_name}")
                continue

            # Read CSV data
            df = pd.read_csv(file_path)
            u_values = df['U Value'].values
            
            num_dofs = len(u_values)
            num_nodes = num_dofs // 6
            
            if num_dofs % 6 != 0:
                print(f"Invalid DOF count in {file_path}: {num_dofs} (not divisible by 6)")
                continue
            if num_nodes != len(mesh_coords):
                print(f"Node count mismatch: Mesh has {len(mesh_coords)} nodes, "
                      f"FEA has {num_nodes} nodes in {file_path}")
                continue

            # Extract u_y (DOF index 1) and theta_z (DOF index 5)
            u_y = u_values[1::6]        # DOF indices: 1, 7, 13, ...
            theta_z = u_values[5::6]    # DOF indices: 5, 11, 17, ...
            
            # Get actual node positions sorted by node ID
            node_ids = sorted(mesh_coords.keys())
            x_vals = np.array([mesh_coords[nid][0] for nid in node_ids])
            
            # Compute derivatives using actual node spacing
            Mx = np.zeros(num_nodes)
            Vx = np.zeros(num_nodes)
            
            # First pass: Compute second derivative (moment)
            for i in range(num_nodes):
                if i == 0:
                    # Forward difference (3 points)
                    dx1 = x_vals[1] - x_vals[0]
                    dx2 = x_vals[2] - x_vals[0]
                    Mx[i] = E * I * (2*u_y[2]/(dx1*dx2) - 2*u_y[1]/(dx1*dx1) + 2*u_y[0]/(dx1*dx2))
                elif i == num_nodes-1:
                    # Backward difference (3 points)
                    dx1 = x_vals[-1] - x_vals[-2]
                    dx2 = x_vals[-1] - x_vals[-3]
                    Mx[i] = E * I * (2*u_y[-3]/(dx1*dx2) - 2*u_y[-2]/(dx1*dx1) + 2*u_y[-1]/(dx1*dx2))
                else:
                    # Central difference (variable spacing)
                    dx_prev = x_vals[i] - x_vals[i-1]
                    dx_next = x_vals[i+1] - x_vals[i]
                    d2u = 2 * (dx_prev*u_y[i+1] - (dx_prev+dx_next)*u_y[i] + dx_next*u_y[i-1]) 
                    d2u /= (dx_prev * dx_next * (dx_prev + dx_next))
                    Mx[i] = E * I * d2u

            # Second pass: Compute first derivative (shear)
            for i in range(num_nodes):
                if i == 0:
                    Vx[i] = (Mx[1] - Mx[0]) / (x_vals[1] - x_vals[0])
                elif i == num_nodes-1:
                    Vx[i] = (Mx[-1] - Mx[-2]) / (x_vals[-1] - x_vals[-2])
                else:
                    # Central difference
                    Vx[i] = (Mx[i+1] - Mx[i-1]) / (x_vals[i+1] - x_vals[i-1])

            results[job_name] = {
                "x":       x_vals,
                "u_y":     u_y,
                "theta_z": theta_z,
                "M_x":     Mx,
                "V_x":     Vx
            }

        except Exception as e:
            print(f"Error processing {file_path}\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    return results

# Dictionary for unit conversions and label text in LaTeX
plot_info = {
    "deflection": {
        "unit_factor": 1000.0,             # [m] → [mm]
        "label_name":  r"$u_{y}(x)\,[mm]$"
    },
    "rotation": {
        "unit_factor": (180.0 / np.pi),    # [rad] → [deg]
        "label_name":  r"$\theta_{z}(x)\,[^\circ]$"
    },
    "shear": {
        "unit_factor": 1.0 / 1000.0,       # [N] → [kN]
        "label_name":  r"$V(x)\,[kN]$"
    },
    "moment": {
        "unit_factor": 1.0 / 1000.0,       # [N·m] → [kN·m]
        "label_name":  r"$M(x)\,[kN \cdot m]$"
    },
}

# -------------------------------------------------------------------------
#   HELPERS TO DETERMINE POINT VS. DISTRIBUTED
# -------------------------------------------------------------------------
POINT_TYPES = {"end", "mid", "quarter"}
DIST_TYPES  = {"Constant", "Quadratic", "Parabolic"}

def get_load_description(load_type):
    """
    Returns something like:
      "Point @ End"
      or
      "Distributed @ Constant"
    depending on the load_type.
    """
    if load_type in POINT_TYPES:
        return f"Point @ {load_type.capitalize()}"
    elif load_type in DIST_TYPES:
        return f"Distributed @ {load_type.capitalize()}"
    else:
        return load_type  # fallback

# -------------------------------------------------------------------------
#   SINGLE CSV WRITER - ALL JOBS IN ONE FILE
# -------------------------------------------------------------------------
def write_all_comparisons_csv(all_job_data):
    """
    Writes a single CSV file "all_jobs_comparison.csv" in SAVE_DIR
    containing blocks for each job.
    """
    out_csv = os.path.join(SAVE_DIR, "NEW_all_jobs_comparison.csv")
    with open(out_csv, "w") as f:
        for job_name, block in all_job_data.items():
            job_title = block["job_title"]  # e.g. "job_0001 / Point @ End"
            f.write(job_title + "\n")
            f.write("x(m),u_y_roark(mm),u_y_fea(mm),theta_roark(deg),theta_fea(deg),V_roark(kN),V_fea(kN),M_roark(kN*m),M_fea(kN*m)\n")

            # Now write each row
            N = len(block["x"])
            for i in range(N):
                row_str = (
                    f"{block['x'][i]:.5g},"
                    f"{block['u_y_roark'][i]:.5g},{block['u_y_fea'][i]:.5g},"
                    f"{block['theta_roark'][i]:.5g},{block['theta_fea'][i]:.5g},"
                    f"{block['V_roark'][i]:.5g},{block['V_fea'][i]:.5g},"
                    f"{block['M_roark'][i]:.5g},{block['M_fea'][i]:.5g}\n"
                )
                f.write(row_str)
            f.write("\n")  # blank line after each job block

    print(f"--> Single CSV file written: {out_csv}")

# -------------------------------------------------------------------------
#   POINT LOAD PLOTTING (accumulates data for CSV)
# -------------------------------------------------------------------------
def plot_comparison_point_loads(point_jobs, job_results, L, E, I, P, all_job_data):
    """
    Creates a single figure with 4 rows x N columns, 
    for the point-load jobs (end, mid, quarter).
    """
    ncols = len(point_jobs)
    if ncols == 0:
        return
        
    fig, axes = plt.subplots(nrows=4, ncols=ncols, figsize=(5*ncols, 12), sharex=True)

    # We'll define x_roark in fine steps for smooth curves
    x_roark = np.linspace(0, L, 300)

    for col_idx, (job_name, load_type) in enumerate(point_jobs.items()):
        # --- Compute Roark's solution at x_roark
        roark_resp = roark_point_load_response(x_roark, L, E, I, P, load_type)
        roark_u  = convert_data("deflection", roark_resp["deflection"])
        roark_th = convert_data("rotation",   roark_resp["rotation"])
        roark_V  = convert_data("shear",      roark_resp["shear"])
        roark_M  = convert_data("moment",     roark_resp["moment"])

        # --- Extract FEA results ---
        data   = job_results[job_name]
        x_fea  = data["x"]
        u_fea  = convert_data("deflection", data["u_y"])
        th_fea = convert_data("rotation",   data["theta_z"])
        V_fea  = convert_data("shear",      data["V_x"])
        M_fea  = convert_data("moment",     data["M_x"])

        # --- Interpolate Roark onto x_fea for CSV block
        roark_u_fea  = np.interp(x_fea, x_roark, roark_u)
        roark_th_fea = np.interp(x_fea, x_roark, roark_th)
        roark_V_fea  = np.interp(x_fea, x_roark, roark_V)
        roark_M_fea  = np.interp(x_fea, x_roark, roark_M)

        # Build a nice label
        job_title = f"{job_name} / {get_load_description(load_type)}"

        # Store data for the final CSV
        all_job_data[job_name] = {
            "job_title": job_title,
            "x": x_fea,
            "u_y_roark": roark_u_fea,
            "u_y_fea":   u_fea,
            "theta_roark": roark_th_fea,
            "theta_fea":   th_fea,
            "V_roark": roark_V_fea,
            "V_fea":    V_fea,
            "M_roark": roark_M_fea,
            "M_fea":    M_fea
        }

        # --- Plot each row ---
        # Create subplot axes depending on column count
        ax0 = axes[0, col_idx] if ncols > 1 else axes[0]
        ax1 = axes[1, col_idx] if ncols > 1 else axes[1]
        ax2 = axes[2, col_idx] if ncols > 1 else axes[2]
        ax3 = axes[3, col_idx] if ncols > 1 else axes[3]

        # Row 0 => deflection
        ax0.plot(x_roark, roark_u, color=COLORS["deflection"], label="Roark", linewidth=2)
        ax0.plot(x_fea, u_fea, 'o', color=COLORS["fea"], label="FEA", markersize=4)
        ax0.set_ylabel(plot_info["deflection"]["label_name"])
        ax0.set_title(job_title)
        ax0.grid(True, linestyle='--', alpha=0.7)
        if col_idx == 0:
            ax0.legend()

        # Row 1 => rotation
        ax1.plot(x_roark, roark_th, color=COLORS["rotation"], linewidth=2)
        ax1.plot(x_fea, th_fea, 'o', color=COLORS["fea"], markersize=4)
        ax1.set_ylabel(plot_info["rotation"]["label_name"])
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Row 2 => shear
        ax2.plot(x_roark, roark_V, color=COLORS["shear"], linewidth=2)
        ax2.plot(x_fea, V_fea, 'o', color=COLORS["fea"], markersize=4)
        ax2.set_ylabel(plot_info["shear"]["label_name"])
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Row 3 => moment
        ax3.plot(x_roark, roark_M, color=COLORS["moment"], linewidth=2)
        ax3.plot(x_fea, M_fea, 'o', color=COLORS["fea"], markersize=4)
        ax3.set_ylabel(plot_info["moment"]["label_name"])
        ax3.set_xlabel(r"$x\,[m]$")
        ax3.grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()
    out_name = os.path.join(SAVE_DIR, "NEW_comparison_point_loads.png")
    fig.savefig(out_name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Point-load comparison figure saved to: {out_name}")

# -------------------------------------------------------------------------
#   DISTRIBUTED LOAD PLOTTING (accumulates data for CSV)
# -------------------------------------------------------------------------
def plot_comparison_distributed_loads(dist_jobs, job_results, L, E, I, w, all_job_data):
    """
    Creates a single figure with 4 rows x N columns, 
    for the distributed-load jobs.
    """
    ncols = len(dist_jobs)
    if ncols == 0:
        return
        
    fig, axes = plt.subplots(nrows=4, ncols=ncols, figsize=(5*ncols, 12), sharex=True)

    # Prepare x-values for Roark
    x_roark = np.linspace(0, L, 300)

    # For mapping your "Constant" → "udl", etc.
    dist_map = {
        "Constant":   "udl",
        "Quadratic":  "triangular",
        "Parabolic":  "parabolic"
    }

    for col_idx, (job_name, load_type) in enumerate(dist_jobs.items()):
        # --- Roark's solution ---
        roark_type = dist_map.get(load_type, "udl")
        roark_resp = roark_distributed_load_response(x_roark, L, E, I, w, roark_type)
        
        roark_u  = convert_data("deflection", roark_resp["deflection"])
        roark_th = convert_data("rotation",   roark_resp["rotation"])
        roark_V  = convert_data("shear",      roark_resp["shear"])
        roark_M  = convert_data("moment",     roark_resp["moment"])

        # --- FEA results ---
        data   = job_results[job_name]
        x_fea  = data["x"]
        u_fea  = convert_data("deflection", data["u_y"])
        th_fea = convert_data("rotation",   data["theta_z"])
        V_fea  = convert_data("shear",      data["V_x"])
        M_fea  = convert_data("moment",     data["M_x"])

        # --- Interpolate Roark onto x_fea
        roark_u_fea  = np.interp(x_fea, x_roark, roark_u)
        roark_th_fea = np.interp(x_fea, x_roark, roark_th)
        roark_V_fea  = np.interp(x_fea, x_roark, roark_V)
        roark_M_fea  = np.interp(x_fea, x_roark, roark_M)

        # Build a nice label
        job_title = f"{job_name} / {get_load_description(load_type)}"

        # Store data for the final CSV
        all_job_data[job_name] = {
            "job_title": job_title,
            "x": x_fea,
            "u_y_roark": roark_u_fea,
            "u_y_fea":   u_fea,
            "theta_roark": roark_th_fea,
            "theta_fea":   th_fea,
            "V_roark": roark_V_fea,
            "V_fea":    V_fea,
            "M_roark": roark_M_fea,
            "M_fea":    M_fea
        }

        # --- Plot each row ---
        # Create subplot axes depending on column count
        ax0 = axes[0, col_idx] if ncols > 1 else axes[0]
        ax1 = axes[1, col_idx] if ncols > 1 else axes[1]
        ax2 = axes[2, col_idx] if ncols > 1 else axes[2]
        ax3 = axes[3, col_idx] if ncols > 1 else axes[3]

        # Row 0 => deflection
        ax0.plot(x_roark, roark_u, color=COLORS["deflection"], label="Roark", linewidth=2)
        ax0.plot(x_fea, u_fea, 'o', color=COLORS["fea"], label="FEA", markersize=4)
        ax0.set_ylabel(plot_info["deflection"]["label_name"])
        ax0.set_title(job_title)
        ax0.grid(True, linestyle='--', alpha=0.7)
        if col_idx == 0:
            ax0.legend()

        # Row 1 => rotation
        ax1.plot(x_roark, roark_th, color=COLORS["rotation"], linewidth=2)
        ax1.plot(x_fea, th_fea, 'o', color=COLORS["fea"], markersize=4)
        ax1.set_ylabel(plot_info["rotation"]["label_name"])
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Row 2 => shear
        ax2.plot(x_roark, roark_V, color=COLORS["shear"], linewidth=2)
        ax2.plot(x_fea, V_fea, 'o', color=COLORS["fea"], markersize=4)
        ax2.set_ylabel(plot_info["shear"]["label_name"])
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Row 3 => moment
        ax3.plot(x_roark, roark_M, color=COLORS["moment"], linewidth=2)
        ax3.plot(x_fea, M_fea, 'o', color=COLORS["fea"], markersize=4)
        ax3.set_ylabel(plot_info["moment"]["label_name"])
        ax3.set_xlabel(r"$x\,[m]$")
        ax3.grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()
    out_name = os.path.join(SAVE_DIR, "NEW_comparison_distributed_loads.png")
    fig.savefig(out_name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Distributed-load comparison figure saved to: {out_name}")

# -------------------------------------------------------------------------
#   MAIN
# -------------------------------------------------------------------------
def main():
    # Load mesh coordinates
    MESH_FILE = "jobs/job_0001/mesh.txt"  # UPDATE THIS PATH
    mesh_coords = parse_mesh(MESH_FILE)
    beam_length = max(x for x, _, _ in mesh_coords.values())
    
    # Material properties
    E = 2.0e11      # [Pa] - Young's modulus
    I = 1.6667e-5   # [m⁴] - Second moment of area

    # Load magnitudes
    P = 100000  # Point load [N]
    w = 100000  # Distributed load [N/m]

    # job -> loadtype mapping
    job_to_loadtype = {
        "job_0001": "end",
        "job_0002": "mid",
        "job_0003": "quarter",
        "job_0004": "Constant",
        "job_0005": "Quadratic",
        "job_0006": "Parabolic"
    }

    # Updated job_files with new paths
    job_files = [
        "post_processing/results/job_0001_2025-06-25_16-21-53-698001_pid45564_c7ad53d9/primary_results/08_U_global.csv",
        "post_processing/results/job_0002_2025-06-25_16-21-53-713139_pid45564_34b7c4b5/primary_results/08_U_global.csv",
        "post_processing/results/job_0003_2025-06-25_16-21-53-717564_pid45564_5fc0e64a/primary_results/08_U_global.csv",
        "post_processing/results/job_0004_2025-06-25_16-21-53-729142_pid45564_5653461a/primary_results/08_U_global.csv",
        "post_processing/results/job_0005_2025-06-25_16-21-53-729142_pid45564_e2590c39/primary_results/08_U_global.csv",
        "post_processing/results/job_0006_2025-06-25_16-21-53-745858_pid45564_009ab639/primary_results/08_U_global.csv",
    ]

    # 1) Extract raw FEA data using actual node positions
    job_results = extract_and_compute_quantities(job_files, mesh_coords, E, I, job_to_loadtype)

    # 2) Separate point-load vs distributed-load jobs
    point_jobs = {}
    dist_jobs  = {}
    for job_name, load_type in job_to_loadtype.items():
        if job_name not in job_results:
            print(f"Warning: Skipping missing job {job_name}")
            continue
        if load_type in POINT_TYPES:
            point_jobs[job_name] = load_type
        elif load_type in DIST_TYPES:
            dist_jobs[job_name]  = load_type
        else:
            print(f"WARNING: {job_name} has unknown load type '{load_type}'")

    # 3) Accumulate data for all jobs in one dict for the final CSV
    all_job_data = {}

    # 4) Plot + accumulate point-load jobs
    if point_jobs:
        plot_comparison_point_loads(point_jobs, job_results, beam_length, E, I, P, all_job_data)

    # 5) Plot + accumulate distributed-load jobs
    if dist_jobs:
        plot_comparison_distributed_loads(dist_jobs, job_results, beam_length, E, I, w, all_job_data)

    # 6) Write a single CSV with all the job blocks
    if all_job_data:
        write_all_comparisons_csv(all_job_data)
    else:
        print("Warning: No job data available for CSV output")

if __name__ == "__main__":
    main()
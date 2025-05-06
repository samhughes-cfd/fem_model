import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
#  ROARK IMPORTS (adjust if needed)
# -------------------------------------------------------------------------
from roarks_formulas_point import roark_point_load_response
from roarks_formulas_distributed import roark_distributed_load_response

# -------------------------------------------------------------------------
#  DIRECTORIES & CONSTANTS
# -------------------------------------------------------------------------
SAVE_DIR = "post_processing/validation_visualisers/deflection_tables/roarks_formulas"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------------------------------------------------
#  COLOR SCHEME (restoring Roark's original palette)
# -------------------------------------------------------------------------
COLORS = {
    "deflection": "#4F81BD",  # Blue
    "rotation":   "#4F81BD",  # Blue
    "shear":      "#9BBB59",  # Green
    "moment":     "#C0504D",  # Red
    "fea":        "black",    # FEA data in black
}

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

def convert_data(category, data_array):
    """Applies a scale factor from `plot_info` to the data array."""
    factor = plot_info[category]["unit_factor"]
    return factor * data_array

# -------------------------------------------------------------------------
#   EXTRACT & COMPUTE QUANTITIES FROM FEA JOB FILES
# -------------------------------------------------------------------------
def extract_and_compute_quantities(job_files, L, E, I):
    """
    Reads each job-file and extracts an approximation for:
       u_y, theta_z, V, M
    from the nodal displacements (assuming each node has 6 DOFs in order: 
    [u_x, u_y, u_z, rot_x, rot_y, rot_z]).

    Returns a dict:
       results = {
         'job_0001': {
           'x': [...],       # in [m]
           'u_y': [...],     # in [m]
           'theta_z': [...], # in [rad]
           'V_x': [...],     # in [N]
           'M_x': [...],     # in [N·m]
         },
         ...
       }
    """
    results = {}

    for file_path in job_files:
        base_name = os.path.basename(file_path)
        job_name  = base_name.split("_static")[0]  # e.g. "job_0001"

        try:
            with open(file_path, "r") as f:
                lines = f.readlines()

            # Filter out commented or empty lines
            data_lines = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]

            # Convert each line to a float
            all_values = np.array([float(x) for x in data_lines], dtype=float)

            # Number of nodes
            num_nodes = len(all_values) // 6
            if num_nodes < 2:
                print(f"Warning: {job_name} has <2 nodes!")
                continue

            # Build x-array from 0..L
            x_vals = np.linspace(0, L, num_nodes)

            # Extract the DOFs for each node
            uy = np.array([all_values[6*i + 1] for i in range(num_nodes)], dtype=float)

            # Approximate rotation, moment, shear via finite differences
            dx = x_vals[1] - x_vals[0]
            theta_z = np.zeros(num_nodes)
            Mx      = np.zeros(num_nodes)
            Vx      = np.zeros(num_nodes)

            for i in range(1, num_nodes - 1):
                # slope ~ d(u_y)/dx
                theta_z[i] = (uy[i+1] - uy[i-1]) / (2*dx)
                # moment ~ E*I * d^2(u_y)/dx^2
                d2 = (uy[i+1] - 2*uy[i] + uy[i-1]) / (dx**2)
                Mx[i] = E * I * d2
                # shear ~ derivative of M wrt x
                Vx[i] = (Mx[i+1] - Mx[i-1]) / (2*dx) if i < (num_nodes-1) else 0.0

            results[job_name] = {
                "x":       x_vals,
                "u_y":     uy,
                "theta_z": theta_z,
                "M_x":     Mx,
                "V_x":     Vx
            }

        except Exception as e:
            print(f"Error reading file: {file_path}\n{e}")

    return results

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
    containing blocks for each job. Each block looks like:

        job_0001 / Point @ End
        x(m),u_y_roark(mm),u_y_fea(mm),theta_roark(deg),theta_fea(deg),V_roark(kN),V_fea(kN),M_roark(kN*m),M_fea(kN*m)
        0,  ...
        0.1,...
        ...
        [blank line]

    where the '...' lines have data for both Roark and FEA, 
    at the x-values in the FEA grid.
    """
    out_csv = os.path.join(SAVE_DIR, "all_jobs_comparison.csv")
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
    Overlays Roark's formula vs. FEA data.
    Accumulates data for CSV writing.
    """
    ncols = len(point_jobs)
    fig, axes = plt.subplots(nrows=4, ncols=ncols, figsize=(5*ncols, 12), sharex=True)

    # We'll define x_roark in fine steps
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
        u_fea  = data["u_y"]     * 1000.0          # [m->mm]
        th_fea = data["theta_z"] * (180.0/np.pi)   # [rad->deg]
        V_fea  = data["V_x"]     / 1000.0          # [N->kN]
        M_fea  = data["M_x"]     / 1000.0          # [N·m->kN·m]

        # --- Interpolate Roark onto x_fea for CSV block
        roark_u_fea  = np.interp(x_fea, x_roark, roark_u)
        roark_th_fea = np.interp(x_fea, x_roark, roark_th)
        roark_V_fea  = np.interp(x_fea, x_roark, roark_V)
        roark_M_fea  = np.interp(x_fea, x_roark, roark_M)

        # Build a nice label like: "job_0001 / Point @ End"
        job_title = f"{job_name} / {get_load_description(load_type)}"

        # Store data in a block for the final CSV
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

        # --- Plot each row
        # Row 0 => deflection
        ax0 = axes[0, col_idx] if ncols>1 else axes[0]
        ax0.plot(x_roark, roark_u, color=COLORS["deflection"], label="Roark")
        ax0.plot(x_fea,   u_fea,   marker='o', color=COLORS["fea"], label="FEA")
        ax0.set_ylabel(plot_info["deflection"]["label_name"])
        ax0.set_title(job_title)
        ax0.grid(True)
        ax0.legend()

        # Row 1 => rotation
        ax1 = axes[1, col_idx] if ncols>1 else axes[1]
        ax1.plot(x_roark, roark_th, color=COLORS["rotation"])
        ax1.plot(x_fea,   th_fea,   marker='o', color=COLORS["fea"])
        ax1.set_ylabel(plot_info["rotation"]["label_name"])
        ax1.grid(True)

        # Row 2 => shear
        ax2 = axes[2, col_idx] if ncols>1 else axes[2]
        ax2.plot(x_roark, roark_V, color=COLORS["shear"])
        ax2.plot(x_fea,   V_fea,   marker='o', color=COLORS["fea"])
        ax2.set_ylabel(plot_info["shear"]["label_name"])
        ax2.grid(True)

        # Row 3 => moment
        ax3 = axes[3, col_idx] if ncols>1 else axes[3]
        ax3.plot(x_roark, roark_M, color=COLORS["moment"])
        ax3.plot(x_fea,   M_fea,   marker='o', color=COLORS["fea"])
        ax3.set_ylabel(plot_info["moment"]["label_name"])
        ax3.set_xlabel(r"$x\,[m]$")
        ax3.grid(True)

    fig.tight_layout()
    out_name = os.path.join(SAVE_DIR, "comparison_point_loads.png")
    fig.savefig(out_name, dpi=150)
    plt.show()
    print(f"Point-load comparison figure saved to: {out_name}")

# -------------------------------------------------------------------------
#   DISTRIBUTED LOAD PLOTTING (accumulates data for CSV)
# -------------------------------------------------------------------------
def plot_comparison_distributed_loads(dist_jobs, job_results, L, E, I, w, all_job_data):
    """
    Creates a single figure with 4 rows x N columns, 
    for the distributed-load jobs (Constant, Quadratic, Parabolic).
    Overlays Roark's formula vs. FEA data.
    Accumulates data for CSV writing.
    """
    ncols = len(dist_jobs)
    fig, axes = plt.subplots(nrows=4, ncols=ncols, figsize=(5*ncols, 12), sharex=True)

    # Prepare x-values for Roark
    x_roark = np.linspace(0, L, 300)

    # For mapping your “Constant” → “udl”, etc.
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
        u_fea  = data["u_y"]     * 1000.0         # [m->mm]
        th_fea = data["theta_z"] * (180.0/np.pi)  # [rad->deg]
        V_fea  = data["V_x"]     / 1000.0         # [N->kN]
        M_fea  = data["M_x"]     / 1000.0         # [N·m->kN·m]

        # --- Interpolate Roark onto x_fea
        roark_u_fea  = np.interp(x_fea, x_roark, roark_u)
        roark_th_fea = np.interp(x_fea, x_roark, roark_th)
        roark_V_fea  = np.interp(x_fea, x_roark, roark_V)
        roark_M_fea  = np.interp(x_fea, x_roark, roark_M)

        # Build a nice label like: "job_0004 / Distributed @ Constant"
        job_title = f"{job_name} / {get_load_description(load_type)}"

        # Store data in a block for the final CSV
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

        # --- Plot each row
        # Row 0 => deflection
        ax0 = axes[0, col_idx] if ncols>1 else axes[0]
        ax0.plot(x_roark, roark_u, color=COLORS["deflection"], label="Roark")
        ax0.plot(x_fea,   u_fea,   marker='o', color=COLORS["fea"], label="FEA")
        ax0.set_ylabel(plot_info["deflection"]["label_name"])
        ax0.set_title(job_title)
        ax0.grid(True)
        ax0.legend()

        # Row 1 => rotation
        ax1 = axes[1, col_idx] if ncols>1 else axes[1]
        ax1.plot(x_roark, roark_th, color=COLORS["rotation"])
        ax1.plot(x_fea,   th_fea,   marker='o', color=COLORS["fea"])
        ax1.set_ylabel(plot_info["rotation"]["label_name"])
        ax1.grid(True)

        # Row 2 => shear
        ax2 = axes[2, col_idx] if ncols>1 else axes[2]
        ax2.plot(x_roark, roark_V, color=COLORS["shear"])
        ax2.plot(x_fea,   V_fea,   marker='o', color=COLORS["fea"])
        ax2.set_ylabel(plot_info["shear"]["label_name"])
        ax2.grid(True)

        # Row 3 => moment
        ax3 = axes[3, col_idx] if ncols>1 else axes[3]
        ax3.plot(x_roark, roark_M, color=COLORS["moment"])
        ax3.plot(x_fea,   M_fea,   marker='o', color=COLORS["fea"])
        ax3.set_ylabel(plot_info["moment"]["label_name"])
        ax3.set_xlabel(r"$x\,[m]$")
        ax3.grid(True)

    fig.tight_layout()
    out_name = os.path.join(SAVE_DIR, "comparison_distributed_loads.png")
    fig.savefig(out_name, dpi=150)
    plt.show()
    print(f"Distributed-load comparison figure saved to: {out_name}")

# -------------------------------------------------------------------------
#   MAIN
# -------------------------------------------------------------------------
def main():
    # Beam parameters
    L = 2           # [m]
    E = 2.0e11        # [Pa]
    I = 1.6667e-5       # [m^4]

    # Single load magnitude for point load
    P = 100000  # [N]

    # Single distributed load magnitude
    w = 100000  # [N/m]

    # job -> loadtype mapping
    job_to_loadtype = {
        "job_0001": "end",
        "job_0002": "mid",
        "job_0003": "quarter",
        "job_0004": "Constant",
        "job_0005": "Quadratic",
        "job_0006": "Parabolic"
    }

    # Example job_files
    job_files = [
        "post_processing/results/job_0001_2025-05-06_15-47-53/primary_results/job_0001_static_global_U_global_2025-05-06_15-47-56.txt",
        "post_processing/results/job_0002_2025-05-06_15-47-53/primary_results/job_0002_static_global_U_global_2025-05-06_15-47-56.txt",
        "post_processing/results/job_0003_2025-05-06_15-47-53/primary_results/job_0003_static_global_U_global_2025-05-06_15-47-56.txt",
        "post_processing/results/job_0004_2025-05-06_15-47-53/primary_results/job_0004_static_global_U_global_2025-05-06_15-47-59.txt",
        "post_processing/results/job_0005_2025-05-06_15-47-53/primary_results/job_0005_static_global_U_global_2025-05-06_15-47-59.txt",
        "post_processing/results/job_0006_2025-05-06_15-47-53/primary_results/job_0006_static_global_U_global_2025-05-06_15-47-59.txt",
    ]

    # 1) Extract raw FEA data
    job_results = extract_and_compute_quantities(job_files, L, E, I)

    # 2) Separate point-load vs distributed-load jobs
    point_jobs = {}
    dist_jobs  = {}
    for job_name, load_type in job_to_loadtype.items():
        if job_name not in job_results:
            continue  # skip if no data
        if load_type in POINT_TYPES:
            point_jobs[job_name] = load_type
        elif load_type in DIST_TYPES:
            dist_jobs[job_name]  = load_type
        else:
            print(f"WARNING: {job_name} has unknown load type '{load_type}'")

    # 3) We'll accumulate data for all jobs in one dict for the final CSV
    all_job_data = {}

    # 4) Plot + accumulate point-load jobs
    if point_jobs:
        plot_comparison_point_loads(point_jobs, job_results, L, E, I, P, all_job_data)

    # 5) Plot + accumulate distributed-load jobs
    if dist_jobs:
        plot_comparison_distributed_loads(dist_jobs, job_results, L, E, I, w, all_job_data)

    # 6) Write a single CSV with all the job blocks
    write_all_comparisons_csv(all_job_data)

if __name__ == "__main__":
    main()

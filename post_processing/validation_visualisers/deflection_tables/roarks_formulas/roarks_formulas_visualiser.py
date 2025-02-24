# post_processing\validation_visualisers\deflection_tables\roarks_formulas_visualiser.py

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
#   IMPORTS FROM YOUR ROARK SCRIPTS
#   (Adjust file paths to match your local folder structure)
# ------------------------------------------------------------------------------
from roarks_formulas_point import roark_point_load_response
from roarks_formulas_distributed import roark_distributed_load_response

# figure save directory

import os
# Define the directory path
SAVE_DIR = "post_processing/validation_visualisers/deflection_tables/roarks_formulas"
# Ensure the directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
#   AESTHETIC STYLES / UNIT CONVERSIONS
# ------------------------------------------------------------------------------
colors = {
    "intensity":  "#7F7F7F",  # gray
    "deflection": "#4F81BD",  # blue
    "rotation":   "#4F81BD",  # also blue
    "shear":      "#9BBB59",  # green
    "moment":     "#C0504D",  # red
}

# Dictionary for unit conversions and label text
plot_info = {
    "intensity": {
        "unit_factor": 1.0,       # keep in [N] for point loads, [N/m] for distributed
        "unit_name":   "(N or N/m)",
        "label_name":  r"$q(x)$",
    },
    "deflection": {
        "unit_factor": 1000.0,    # m -> mm
        "unit_name":   "mm",
        "label_name":  r"$u_y(x)$",
    },
    "rotation": {
        "unit_factor": 180.0/np.pi,  # rad -> degrees
        "unit_name":   "°",
        "label_name":  r"$\theta_z(x)$",
    },
    "shear": {
        "unit_factor": 1.0/1000.0,  # N -> kN
        "unit_name":   "kN",
        "label_name":  r"$V(x)$",
    },
    "moment": {
        "unit_factor": 1.0/1000.0,  # N·m -> kN·m
        "unit_name":   "kN·m",
        "label_name":  r"$M(x)$",
    },
}

def convert_data(category, data):
    """Applies the scale factor from plot_info to 'data'."""
    factor = plot_info[category]["unit_factor"]
    return factor * data


# ------------------------------------------------------------------------------
#   FIGURE 1: POINT LOADS => 3 rows: [q, u, theta], 3 cols: [end, mid, quarter]
# ------------------------------------------------------------------------------
def plot_point_load_fig1(L, E, I, P):
    """
    Figure 1 => 3×3 subplots:
      Rows =  q(x),  deflection u_y,  rotation theta
      Cols = 'end', 'mid', 'quarter'
    """
    x_vals = np.linspace(0, L, 200)
    load_types = ["end", "mid", "quarter"]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    fig.suptitle("Figure 1: Point Loads (q, u, theta)", fontsize=16)

    for col_idx, lt in enumerate(load_types):
        # Response dictionary from roark_point_load_response
        resp = roark_point_load_response(x_vals, L, E, I, P, lt)

        # Build an array that is zero except at x=a => negative spike
        q_spike = resp["intensity"].copy()
        # Determine the load location a
        if lt == "end":
            a = L
        elif lt == "mid":
            a = L / 2
        else:  # "quarter"
            a = L / 4

        # Insert the downward spike at index for x=a
        idx_a = np.argmin(abs(x_vals - a))
        q_spike[idx_a] = -P

        # Extract deflection & rotation
        u_vals = resp["deflection"]  # [m]
        th_vals = resp["rotation"]   # [rad]

        # Row 1 => q(x)
        ax_q = axes[0, col_idx]
        conv_q = convert_data("intensity", q_spike)
        # Instead of stem(), use plot() to show a vertical line from 0 -> conv_q[idx_a]
        # Plot entire array is basically zero except one spike, so let's do:
        spike_x = x_vals[idx_a]
        spike_y = conv_q[idx_a]
        ax_q.plot([spike_x, spike_x], [0, spike_y], color=colors["intensity"], linewidth=2)
        # Optionally mark the tip:
        ax_q.plot(spike_x, spike_y, marker="o", color=colors["intensity"])

        # Labeling
        label_y = f"{plot_info['intensity']['label_name']} [{plot_info['intensity']['unit_name']}]"
        ax_q.set_ylabel(label_y if col_idx == 0 else "")
        ax_q.set_title(f"Point Load @ {lt.capitalize()}", fontsize=13)

        # Row 2 => u(x)
        ax_u = axes[1, col_idx]
        conv_u = convert_data("deflection", u_vals)
        ax_u.plot(x_vals, conv_u, color=colors["deflection"], label="u(x)")
        label_y = f"{plot_info['deflection']['label_name']} [{plot_info['deflection']['unit_name']}]"
        ax_u.set_ylabel(label_y if col_idx == 0 else "")

        # Row 3 => theta(x)
        ax_th = axes[2, col_idx]
        conv_th = convert_data("rotation", th_vals)
        ax_th.plot(x_vals, conv_th, color=colors["rotation"], label="theta(x)")
        label_y = f"{plot_info['rotation']['label_name']} [{plot_info['rotation']['unit_name']}]"
        ax_th.set_ylabel(label_y if col_idx == 0 else "")
        ax_th.set_xlabel("x [m]")

        # Cosmetic formatting
        for row in range(3):
            axes[row, col_idx].grid(True)
            axes[row, col_idx].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.savefig(os.path.join(SAVE_DIR,'point_q_u_theta.png'))
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
#   FIGURE 2: POINT LOADS => 3 rows: [q, V, M], 3 cols: [end, mid, quarter]
# ------------------------------------------------------------------------------
def plot_point_load_fig2(L, E, I, P):
    """
    Figure 2 => 3×3 subplots:
      Rows =  q(x),  shear V(x),  moment M(x)
      Cols = 'end', 'mid', 'quarter'
    """
    x_vals = np.linspace(0, L, 200)
    load_types = ["end", "mid", "quarter"]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    fig.suptitle("Figure 2: Point Loads (q, V, M)", fontsize=16)

    for col_idx, lt in enumerate(load_types):
        resp = roark_point_load_response(x_vals, L, E, I, P, lt)

        # Insert a spike at x=a
        q_spike = resp["intensity"].copy()
        if lt == "end":
            a = L
        elif lt == "mid":
            a = L / 2
        else:  # quarter
            a = L / 4

        idx_a = np.argmin(abs(x_vals - a))
        q_spike[idx_a] = -P

        # Shear & Moment
        V_vals = resp["shear"]
        M_vals = resp["moment"]

        # Row 1 => q(x): use a vertical line
        ax_q = axes[0, col_idx]
        conv_q = convert_data("intensity", q_spike)
        spike_x = x_vals[idx_a]
        spike_y = conv_q[idx_a]
        ax_q.plot([spike_x, spike_x], [0, spike_y], color=colors["intensity"], linewidth=2)
        ax_q.plot(spike_x, spike_y, marker="o", color=colors["intensity"])
        label_y = f"{plot_info['intensity']['label_name']} [{plot_info['intensity']['unit_name']}]"
        ax_q.set_ylabel(label_y if col_idx == 0 else "")
        ax_q.set_title(f"Point Load @ {lt.capitalize()}", fontsize=13)

        # Row 2 => V(x)
        ax_v = axes[1, col_idx]
        conv_v = convert_data("shear", V_vals)
        ax_v.step(x_vals, conv_v, where="post", color=colors["shear"])
        label_y = f"{plot_info['shear']['label_name']} [{plot_info['shear']['unit_name']}]"
        ax_v.set_ylabel(label_y if col_idx == 0 else "")

        # Row 3 => M(x)
        ax_m = axes[2, col_idx]
        conv_m = convert_data("moment", M_vals)
        ax_m.plot(x_vals, conv_m, color=colors["moment"])
        label_y = f"{plot_info['moment']['label_name']} [{plot_info['moment']['unit_name']}]"
        ax_m.set_ylabel(label_y if col_idx == 0 else "")
        ax_m.set_xlabel("x [m]")

        # Cosmetic formatting
        for row in range(3):
            axes[row, col_idx].grid(True)
            axes[row, col_idx].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.savefig(os.path.join(SAVE_DIR,'point_q_V_M.png'))
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
#   FIGURE 3: DISTRIBUTED LOADS => 3 rows: [q, u, theta], 3 cols: [udl, tri, parab]
# ------------------------------------------------------------------------------
def plot_distributed_fig3(L, E, I, w):
    """
    Figure 3 => 3×3 subplots:
      Rows = q(x), u_y(x), theta_z(x)
      Cols = 'udl', 'triangular', 'parabolic'
    """
    x_vals = np.linspace(0, L, 750)
    load_types = ["udl", "triangular", "parabolic"]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    fig.suptitle("Figure 3: Distributed Loads (q, u, theta)", fontsize=16)

    for col_idx, lt in enumerate(load_types):
        resp = roark_distributed_load_response(x_vals, L, E, I, w, lt)

        q_vals = resp["intensity"]      # [N/m]
        u_vals = resp["deflection"]     # [m]
        th_vals = resp["rotation"]      # [rad]

        # Row 1 => q(x) -> continuous line
        ax_q = axes[0, col_idx]
        conv_q = convert_data("intensity", q_vals)
        ax_q.plot(x_vals, conv_q, color=colors["intensity"])
        label_y = f"{plot_info['intensity']['label_name']} [{plot_info['intensity']['unit_name']}]"
        ax_q.set_ylabel(label_y if col_idx == 0 else "")
        ax_q.set_title(f"{lt.capitalize()} Load", fontsize=13)

        # Row 2 => u(x)
        ax_u = axes[1, col_idx]
        conv_u = convert_data("deflection", u_vals)
        ax_u.plot(x_vals, conv_u, color=colors["deflection"])
        label_y = f"{plot_info['deflection']['label_name']} [{plot_info['deflection']['unit_name']}]"
        ax_u.set_ylabel(label_y if col_idx == 0 else "")

        # Row 3 => theta_z(x)
        ax_th = axes[2, col_idx]
        conv_th = convert_data("rotation", th_vals)
        ax_th.plot(x_vals, conv_th, color=colors["rotation"])
        label_y = f"{plot_info['rotation']['label_name']} [{plot_info['rotation']['unit_name']}]"
        ax_th.set_ylabel(label_y if col_idx == 0 else "")
        ax_th.set_xlabel("x [m]")

        for row in range(3):
            axes[row, col_idx].grid(True)
            axes[row, col_idx].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    fig.savefig(os.path.join(SAVE_DIR,'distributed_q_u_theta.png'))
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
#   FIGURE 4: DISTRIBUTED LOADS => 3 rows: [q, V, M], 3 cols: [udl, tri, parab]
# ------------------------------------------------------------------------------
def plot_distributed_fig4(L, E, I, w):
    """
    Figure 4 => 3×3 subplots:
      Rows = q(x), V(x), M(x)
      Cols = 'udl', 'triangular', 'parabolic'
    """
    x_vals = np.linspace(0, L, 200)
    load_types = ["udl", "triangular", "parabolic"]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    fig.suptitle("Figure 4: Distributed Loads (q, V, M)", fontsize=16)

    for col_idx, lt in enumerate(load_types):
        resp = roark_distributed_load_response(x_vals, L, E, I, w, lt)

        q_vals = resp["intensity"]      # [N/m]
        V_vals = resp["shear"]          # [N]
        M_vals = resp["moment"]         # [N·m]

        # Row 1 => q(x)
        ax_q = axes[0, col_idx]
        conv_q = convert_data("intensity", q_vals)
        ax_q.plot(x_vals, conv_q, color=colors["intensity"])
        label_y = f"{plot_info['intensity']['label_name']} [{plot_info['intensity']['unit_name']}]"
        ax_q.set_ylabel(label_y if col_idx == 0 else "")
        ax_q.set_title(f"{lt.capitalize()} Load", fontsize=13)

        # Row 2 => V(x)
        ax_v = axes[1, col_idx]
        conv_v = convert_data("shear", V_vals)
        ax_v.plot(x_vals, conv_v, color=colors["shear"])
        label_y = f"{plot_info['shear']['label_name']} [{plot_info['shear']['unit_name']}]"
        ax_v.set_ylabel(label_y if col_idx == 0 else "")

        # Row 3 => M(x)
        ax_m = axes[2, col_idx]
        conv_m = convert_data("moment", M_vals)
        ax_m.plot(x_vals, conv_m, color=colors["moment"])
        label_y = f"{plot_info['moment']['label_name']} [{plot_info['moment']['unit_name']}]"
        ax_m.set_ylabel(label_y if col_idx == 0 else "")
        ax_m.set_xlabel("x [m]")

        for row in range(3):
            axes[row, col_idx].grid(True)
            axes[row, col_idx].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    fig.savefig(os.path.join(SAVE_DIR,'distributed_q_V_M.png'))
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------------
#   FIGURE 5: POINT LOADS => 4 rows: [u, theta, V, M], 3 cols: [end, mid, quarter]
# ------------------------------------------------------------------------------------

def plot_point_load_fig5(L, E, I, P):
    """
    Plots the response of a beam under point loads:
    - Rows = [Deflection u_y, Rotation theta_z, Shear V, Moment M]
    - Columns = ['end', 'mid', 'quarter']
    """
    x_vals = np.linspace(0, L, 200)
    load_types = ["end", "mid", "quarter"]

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))
    fig.suptitle("Point Load Response: u_y, θ_z, V, M", fontsize=16)

    for col_idx, lt in enumerate(load_types):
        resp = roark_point_load_response(x_vals, L, E, I, P, lt)

        # Extract response values
        u_vals = resp["deflection"]
        th_vals = resp["rotation"]
        V_vals = resp["shear"]
        M_vals = resp["moment"]

        # Row 1 => u_y(x)
        ax_u = axes[0, col_idx]
        ax_u.plot(x_vals, convert_data("deflection", u_vals), color=colors["deflection"])
        ax_u.set_ylabel(f"{plot_info['deflection']['label_name']} [{plot_info['deflection']['unit_name']}]")

        # Row 2 => θ_z(x)
        ax_th = axes[1, col_idx]
        ax_th.plot(x_vals, convert_data("rotation", th_vals), color=colors["rotation"])
        ax_th.set_ylabel(f"{plot_info['rotation']['label_name']} [{plot_info['rotation']['unit_name']}]")

        # Row 3 => V(x)
        ax_v = axes[2, col_idx]
        ax_v.step(x_vals, convert_data("shear", V_vals), where="post", color=colors["shear"])
        ax_v.set_ylabel(f"{plot_info['shear']['label_name']} [{plot_info['shear']['unit_name']}]")

        # Row 4 => M(x)
        ax_m = axes[3, col_idx]
        ax_m.plot(x_vals, convert_data("moment", M_vals), color=colors["moment"])
        ax_m.set_ylabel(f"{plot_info['moment']['label_name']} [{plot_info['moment']['unit_name']}]")
        ax_m.set_xlabel("x [m]")

        # Titles for each column
        ax_u.set_title(f"Point Load @ {lt.capitalize()}", fontsize=13)

        # Formatting
        for row in range(4):
            axes[row, col_idx].grid(True)
            axes[row, col_idx].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    fig.savefig(os.path.join(SAVE_DIR,'point_u_theta_V_M.png'))
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------------
#   FIGURE 6: DISTRIBUTED LOADS => 4 rows: [u, theta, V, M], 3 cols: [udl, tri, parab]
# ------------------------------------------------------------------------------------

def plot_distributed_fig6(L, E, I, w):
    """
    Plots the response of a beam under distributed loads:
    - Rows = [Deflection u_y, Rotation theta_z, Shear V, Moment M]
    - Columns = ['UDL', 'Triangular', 'Parabolic']
    """
    x_vals = np.linspace(0, L, 750)
    load_types = ["udl", "triangular", "parabolic"]

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))
    fig.suptitle("Distributed Load Response: u_y, θ_z, V, M", fontsize=16)

    for col_idx, lt in enumerate(load_types):
        resp = roark_distributed_load_response(x_vals, L, E, I, w, lt)

        # Extract response values
        u_vals = resp["deflection"]
        th_vals = resp["rotation"]
        V_vals = resp["shear"]
        M_vals = resp["moment"]

        # Row 1 => u_y(x)
        ax_u = axes[0, col_idx]
        ax_u.plot(x_vals, convert_data("deflection", u_vals), color=colors["deflection"])
        ax_u.set_ylabel(f"{plot_info['deflection']['label_name']} [{plot_info['deflection']['unit_name']}]")

        # Row 2 => θ_z(x)
        ax_th = axes[1, col_idx]
        ax_th.plot(x_vals, convert_data("rotation", th_vals), color=colors["rotation"])
        ax_th.set_ylabel(f"{plot_info['rotation']['label_name']} [{plot_info['rotation']['unit_name']}]")

        # Row 3 => V(x)
        ax_v = axes[2, col_idx]
        ax_v.plot(x_vals, convert_data("shear", V_vals), color=colors["shear"])
        ax_v.set_ylabel(f"{plot_info['shear']['label_name']} [{plot_info['shear']['unit_name']}]")

        # Row 4 => M(x)
        ax_m = axes[3, col_idx]
        ax_m.plot(x_vals, convert_data("moment", M_vals), color=colors["moment"])
        ax_m.set_ylabel(f"{plot_info['moment']['label_name']} [{plot_info['moment']['unit_name']}]")
        ax_m.set_xlabel("x [m]")

        # Titles for each column
        ax_u.set_title(f"{lt.capitalize()} Load", fontsize=13)

        # Formatting
        for row in range(4):
            axes[row, col_idx].grid(True)
            axes[row, col_idx].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    fig.savefig(os.path.join(SAVE_DIR,'distributed_u_theta_V_M.png'))
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
#   MAIN: CALL ALL SIX PLOTTING FUNCTIONS
# ------------------------------------------------------------------------------
def main():
    """
    Creates 6 separate figures:
      1) Point loads => q, u, theta
      2) Point loads => q, V, M
      3) Distributed => q, u, theta
      4) Distributed => q, V, M
      5) Point loads => u, theta, V, M  
      6) Distributed => u, theta, V, M  
    using normal `plot()` calls for the point-load spike (no `stem()`).
    """
    # Beam parameters
    L = 8.0           # [m]
    E = 2.0e11        # [Pa]
    I = 2.67e-7       # [m^4]

    # Point load [N]
    P = 1000.0

    # Distributed load [N/m]
    w = 1000.0

    # (1) Figure 1: point loads => q, u, theta
    plot_point_load_fig1(L, E, I, P)

    # (2) Figure 2: point loads => q, V, M
    plot_point_load_fig2(L, E, I, P)

    # (3) Figure 3: distributed => q, u, theta
    plot_distributed_fig3(L, E, I, w)

    # (4) Figure 4: distributed => q, V, M
    plot_distributed_fig4(L, E, I, w)

    # (5) Figure 5: point loads => u, theta, V, M
    plot_point_load_fig5(L, E, I, P)

    # (6) Figure 6: distributed loads => u, theta, V, M
    plot_distributed_fig6(L, E, I, w)


if __name__ == "__main__":
    main()

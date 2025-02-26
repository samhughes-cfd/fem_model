# post_processing\validation_visualisers\deflection_tables\roarks_formulas_visualiser.py

import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------------------------
#   IMPORTS FROM YOUR ROARK SCRIPTS
#   (Adjust file paths to match your local folder structure)
# ------------------------------------------------------------------------------
from roarks_formulas_point import roark_point_load_response
from roarks_formulas_distributed import roark_distributed_load_response

# Define the directory path
SAVE_DIR = "post_processing/validation_visualisers/deflection_tables/roarks_formulas"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
#   AESTHETIC STYLES / UNIT CONVERSIONS
# ------------------------------------------------------------------------------
colors = {
    "intensity":  "#7F7F7F",  # gray
    "deflection": "#4F81BD",  # blue
    "rotation":   "#4F81BD",  # also blue
    "shear":      "#9BBB59",  # green
    "moment":     "#C0504D",  # red
}

# Dictionary for unit conversions and label text in LaTeX
plot_info = {
    "intensity": {
        "unit_factor": 1.0/1000.0,        # N -> kN or N/m -> kN/m
        "unit_name":   r"$kN \text{ or } kN/m$",
        "label_name":  r"$q(x)$",
    },
    "deflection": {
        "unit_factor": 1000.0,            # m -> mm
        "unit_name":   r"$mm$",
        "label_name":  r"$u_{y}(x)$",
    },
    "rotation": {
        "unit_factor": 180.0/np.pi,       # rad -> degrees
        "unit_name":   r"${}^{\circ}$",
        "label_name":  r"$\theta_{z}(x)$",
    },
    "shear": {
        "unit_factor": 1.0/1000.0,        # N -> kN
        "unit_name":   r"$kN$",
        "label_name":  r"$V(x)$",
    },
    "moment": {
        "unit_factor": 1.0/1000.0,        # N·m -> kN·m
        "unit_name":   r"$kN \cdot m$",
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
    fig.suptitle("Figure 1: Point Loads (q, u, θ)", fontsize=16)

    for col_idx, lt in enumerate(load_types):
        resp = roark_point_load_response(x_vals, L, E, I, P, lt)

        # Build array for a negative spike
        q_spike = resp["intensity"].copy()
        if lt == "end":
            a = L
        elif lt == "mid":
            a = L / 2
        else:
            a = L / 4
        idx_a = np.argmin(abs(x_vals - a))
        q_spike[idx_a] = -P

        # Extract deflection & rotation
        u_vals = resp["deflection"]
        th_vals = resp["rotation"]

        # Row 1 => q(x)
        ax_q = axes[0, col_idx]
        conv_q = convert_data("intensity", q_spike)
        spike_x = x_vals[idx_a]
        spike_y = conv_q[idx_a]
        ax_q.plot([spike_x, spike_x], [0, spike_y], color=colors["intensity"], linewidth=2)
        ax_q.plot(spike_x, spike_y, marker="o", color=colors["intensity"])
        if col_idx == 0:
            ax_q.set_ylabel(
                f"{plot_info['intensity']['label_name']} [{plot_info['intensity']['unit_name']}]"
            )
        ax_q.set_title(f"Point Load @ {lt.capitalize()}", fontsize=13)

        # Row 2 => deflection
        ax_u = axes[1, col_idx]
        conv_u = convert_data("deflection", u_vals)
        ax_u.plot(x_vals, conv_u, color=colors["deflection"])
        if col_idx == 0:
            ax_u.set_ylabel(
                f"{plot_info['deflection']['label_name']} [{plot_info['deflection']['unit_name']}]"
            )

        # Row 3 => rotation
        ax_th = axes[2, col_idx]
        conv_th = convert_data("rotation", th_vals)
        ax_th.plot(x_vals, conv_th, color=colors["rotation"])
        if col_idx == 0:
            ax_th.set_ylabel(
                f"{plot_info['rotation']['label_name']} [{plot_info['rotation']['unit_name']}]"
            )
        ax_th.set_xlabel(r"$x\,(\mathrm{m})$")

        # Cosmetic formatting
        for row in range(3):
            axes[row, col_idx].grid(True)
            axes[row, col_idx].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'point_q_u_theta.png'))
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

        q_spike = resp["intensity"].copy()
        if lt == "end":
            a = L
        elif lt == "mid":
            a = L / 2
        else:
            a = L / 4
        idx_a = np.argmin(abs(x_vals - a))
        q_spike[idx_a] = -P

        V_vals = resp["shear"]
        M_vals = resp["moment"]

        # Row 1 => q(x)
        ax_q = axes[0, col_idx]
        conv_q = convert_data("intensity", q_spike)
        spike_x = x_vals[idx_a]
        spike_y = conv_q[idx_a]
        ax_q.plot([spike_x, spike_x], [0, spike_y], color=colors["intensity"], linewidth=2)
        ax_q.plot(spike_x, spike_y, marker="o", color=colors["intensity"])
        if col_idx == 0:
            ax_q.set_ylabel(
                f"{plot_info['intensity']['label_name']} [{plot_info['intensity']['unit_name']}]"
            )
        ax_q.set_title(f"Point Load @ {lt.capitalize()}", fontsize=13)

        # Row 2 => V(x)
        ax_v = axes[1, col_idx]
        conv_v = convert_data("shear", V_vals)
        ax_v.step(x_vals, conv_v, where="post", color=colors["shear"])
        if col_idx == 0:
            ax_v.set_ylabel(
                f"{plot_info['shear']['label_name']} [{plot_info['shear']['unit_name']}]"
            )

        # Row 3 => M(x)
        ax_m = axes[2, col_idx]
        conv_m = convert_data("moment", M_vals)
        ax_m.plot(x_vals, conv_m, color=colors["moment"])
        if col_idx == 0:
            ax_m.set_ylabel(
                f"{plot_info['moment']['label_name']} [{plot_info['moment']['unit_name']}]"
            )
        ax_m.set_xlabel(r"$x\,(\mathrm{m})$")

        # Cosmetic formatting
        for row in range(3):
            axes[row, col_idx].grid(True)
            axes[row, col_idx].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'point_q_V_M.png'))
    plt.show()


# ------------------------------------------------------------------------------
#   FIGURE 3: DISTRIBUTED LOADS => 3 rows: [q, u, θ], 3 cols: [udl, triangular, parabolic]
# ------------------------------------------------------------------------------
def plot_distributed_fig3(L, E, I, w):
    """
    Figure 3 => 3×3 subplots:
      Rows = q(x), u_y(x), θ_z(x)
      Cols = 'udl', 'triangular', 'parabolic'
    """
    x_vals = np.linspace(0, L, 750)
    load_types = ["udl", "triangular", "parabolic"]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    fig.suptitle("Figure 3: Distributed Loads (q, u, θ)", fontsize=16)

    for col_idx, lt in enumerate(load_types):
        resp = roark_distributed_load_response(x_vals, L, E, I, w, lt)
        q_vals = resp["intensity"]
        u_vals = resp["deflection"]
        th_vals = resp["rotation"]

        # Row 1 => q(x)
        ax_q = axes[0, col_idx]
        conv_q = convert_data("intensity", q_vals)
        ax_q.plot(x_vals, conv_q, color=colors["intensity"])
        if col_idx == 0:
            ax_q.set_ylabel(
                f"{plot_info['intensity']['label_name']} [{plot_info['intensity']['unit_name']}]"
            )
        ax_q.set_title(f"{lt.capitalize()} Load", fontsize=13)

        # Row 2 => u(x)
        ax_u = axes[1, col_idx]
        conv_u = convert_data("deflection", u_vals)
        ax_u.plot(x_vals, conv_u, color=colors["deflection"])
        if col_idx == 0:
            ax_u.set_ylabel(
                f"{plot_info['deflection']['label_name']} [{plot_info['deflection']['unit_name']}]"
            )

        # Row 3 => θ(x)
        ax_th = axes[2, col_idx]
        conv_th = convert_data("rotation", th_vals)
        ax_th.plot(x_vals, conv_th, color=colors["rotation"])
        if col_idx == 0:
            ax_th.set_ylabel(
                f"{plot_info['rotation']['label_name']} [{plot_info['rotation']['unit_name']}]"
            )
        ax_th.set_xlabel(r"$x\,(\mathrm{m})$")

        for row in range(3):
            axes[row, col_idx].grid(True)
            axes[row, col_idx].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'distributed_q_u_theta.png'))
    plt.show()


# ------------------------------------------------------------------------------
#   FIGURE 4: DISTRIBUTED LOADS => 3 rows: [q, V, M], 3 cols: [udl, triangular, parabolic]
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
        q_vals = resp["intensity"]
        V_vals = resp["shear"]
        M_vals = resp["moment"]

        # Row 1 => q(x)
        ax_q = axes[0, col_idx]
        conv_q = convert_data("intensity", q_vals)
        ax_q.plot(x_vals, conv_q, color=colors["intensity"])
        if col_idx == 0:
            ax_q.set_ylabel(
                f"{plot_info['intensity']['label_name']} [{plot_info['intensity']['unit_name']}]"
            )
        ax_q.set_title(f"{lt.capitalize()} Load", fontsize=13)

        # Row 2 => V(x)
        ax_v = axes[1, col_idx]
        conv_v = convert_data("shear", V_vals)
        ax_v.plot(x_vals, conv_v, color=colors["shear"])
        if col_idx == 0:
            ax_v.set_ylabel(
                f"{plot_info['shear']['label_name']} [{plot_info['shear']['unit_name']}]"
            )

        # Row 3 => M(x)
        ax_m = axes[2, col_idx]
        conv_m = convert_data("moment", M_vals)
        ax_m.plot(x_vals, conv_m, color=colors["moment"])
        if col_idx == 0:
            ax_m.set_ylabel(
                f"{plot_info['moment']['label_name']} [{plot_info['moment']['unit_name']}]"
            )
        ax_m.set_xlabel(r"$x\,(\mathrm{m})$")

        for row in range(3):
            axes[row, col_idx].grid(True)
            axes[row, col_idx].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'distributed_q_V_M.png'))
    plt.show()


# ------------------------------------------------------------------------------------
#   FIGURE 5: POINT LOADS => 4 rows: [u, θ, V, M], 3 cols: [end, mid, quarter]
# ------------------------------------------------------------------------------------
def plot_point_load_fig5_1(L, E, I, P):
    """
    Plots the response of a beam under point loads:
    - Rows = [Deflection u_y, Rotation θ_z, Shear V, Moment M]
    - Columns = ['end', 'mid', 'quarter']
    """
    x_vals = np.linspace(0, L, 200)
    load_types = ["end", "mid", "quarter"]

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))
    fig.suptitle(r"Point Load Response: $u_y$, $\theta_z$, $V$, $M$", fontsize=16)

    for col_idx, lt in enumerate(load_types):
        resp = roark_point_load_response(x_vals, L, E, I, P, lt)
        u_vals = resp["deflection"]
        th_vals = resp["rotation"]
        V_vals = resp["shear"]
        M_vals = resp["moment"]

        # Row 1 => deflection
        ax_u = axes[0, col_idx]
        ax_u.plot(x_vals, convert_data("deflection", u_vals), color=colors["deflection"])
        if col_idx == 0:
            ax_u.set_ylabel(
                f"{plot_info['deflection']['label_name']} [{plot_info['deflection']['unit_name']}]"
            )

        # Row 2 => rotation
        ax_th = axes[1, col_idx]
        ax_th.plot(x_vals, convert_data("rotation", th_vals), color=colors["rotation"])
        if col_idx == 0:
            ax_th.set_ylabel(
                f"{plot_info['rotation']['label_name']} [{plot_info['rotation']['unit_name']}]"
            )

        # Row 3 => shear
        ax_v = axes[2, col_idx]
        V_converted = convert_data("shear", V_vals)
        ax_v.step(x_vals, V_converted, where="post", color=colors["shear"])
        ax_v.fill_between(x_vals, V_converted, 0, step="post", color=colors["shear"], alpha=0.25)
        if col_idx == 0:
            ax_v.set_ylabel(
                f"{plot_info['shear']['label_name']} [{plot_info['shear']['unit_name']}]"
            )

        # Row 4 => moment
        ax_m = axes[3, col_idx]
        M_converted = convert_data("moment", M_vals)
        ax_m.plot(x_vals, M_converted, color=colors["moment"])
        ax_m.fill_between(x_vals, M_converted, 0, color=colors["moment"], alpha=0.25)
        if col_idx == 0:
            ax_m.set_ylabel(
                f"{plot_info['moment']['label_name']} [{plot_info['moment']['unit_name']}]"
            )
        ax_m.set_xlabel(r"$x\,(\mathrm{m})$")

        ax_u.set_title(f"Point Load @ {lt.capitalize()}", fontsize=13)

        for row in range(4):
            axes[row, col_idx].grid(True)
            axes[row, col_idx].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'point_u_theta_V_M.png'))
    plt.show()


# ------------------------------------------------------------------------------------
#   FIGURE 6: DISTRIBUTED LOADS => 4 rows: [u, θ, V, M], 3 cols: [udl, triangular, parabolic]
# ------------------------------------------------------------------------------------
def plot_distributed_fig6_1(L, E, I, w):
    """
    Plots the response of a beam under distributed loads:
    - Rows = [Deflection u_y, Rotation θ_z, Shear V, Moment M]
    - Columns = ['udl', 'triangular', 'parabolic']
    """
    x_vals = np.linspace(0, L, 750)
    load_types = ["udl", "triangular", "parabolic"]

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))
    fig.suptitle(r"Distributed Load Response: $u_y$, $\theta_z$, $V$, $M$", fontsize=16)

    for col_idx, lt in enumerate(load_types):
        resp = roark_distributed_load_response(x_vals, L, E, I, w, lt)
        u_vals = resp["deflection"]
        th_vals = resp["rotation"]
        V_vals = resp["shear"]
        M_vals = resp["moment"]

        # Row 1 => deflection
        ax_u = axes[0, col_idx]
        ax_u.plot(x_vals, convert_data("deflection", u_vals), color=colors["deflection"])
        if col_idx == 0:
            ax_u.set_ylabel(
                f"{plot_info['deflection']['label_name']} [{plot_info['deflection']['unit_name']}]"
            )

        # Row 2 => rotation
        ax_th = axes[1, col_idx]
        ax_th.plot(x_vals, convert_data("rotation", th_vals), color=colors["rotation"])
        if col_idx == 0:
            ax_th.set_ylabel(
                f"{plot_info['rotation']['label_name']} [{plot_info['rotation']['unit_name']}]"
            )

        # Row 3 => shear
        ax_v = axes[2, col_idx]
        V_converted = convert_data("shear", V_vals)
        ax_v.plot(x_vals, V_converted, color=colors["shear"])
        ax_v.fill_between(x_vals, V_converted, 0, color=colors["shear"], alpha=0.25)
        if col_idx == 0:
            ax_v.set_ylabel(
                f"{plot_info['shear']['label_name']} [{plot_info['shear']['unit_name']}]"
            )

        # Row 4 => moment
        ax_m = axes[3, col_idx]
        M_converted = convert_data("moment", M_vals)
        ax_m.plot(x_vals, M_converted, color=colors["moment"])
        ax_m.fill_between(x_vals, M_converted, 0, color=colors["moment"], alpha=0.25)
        if col_idx == 0:
            ax_m.set_ylabel(
                f"{plot_info['moment']['label_name']} [{plot_info['moment']['unit_name']}]"
            )
        ax_m.set_xlabel(r"$x\,(\mathrm{m})$")

        ax_u.set_title(f"{lt.capitalize()} Load", fontsize=13)

        for row in range(4):
            axes[row, col_idx].grid(True)
            axes[row, col_idx].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'distributed_u_theta_V_M.png'))
    plt.show()

def plot_load_intensities_2x3(L, E, I, P, w):
    """
    Creates a 2x3 subplot figure for q(x):
      - Row 0 => Point loads: end, mid, quarter (all with q in [kN])
      - Row 1 => Distributed loads: udl, triangular, parabolic (q in [kN/m])
    """
    # x-arrays for plotting
    x_vals_pt = np.linspace(0, L, 200)   # For point loads
    x_vals_dist = np.linspace(0, L, 750) # For distributed loads

    point_load_types = ["end", "mid", "quarter"]
    dist_load_types = ["udl", "triangular", "parabolic"]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), sharex=False)

    # ----------------------
    #  Row 0 => Point Loads
    # ----------------------
    for col_idx, lt in enumerate(point_load_types):
        resp_pt = roark_point_load_response(x_vals_pt, L, E, I, P, lt)

        # Build negative spike
        q_spike = np.zeros_like(x_vals_pt)
        if lt == "end":
            a = L
        elif lt == "mid":
            a = L / 2
        else:  # "quarter"
            a = L / 4
        idx_a = np.argmin(abs(x_vals_pt - a))
        q_spike[idx_a] = -P  # Negative spike

        # Convert from N to kN
        q_vals_pt = convert_data("intensity", q_spike)  # -> [kN]

        ax = axes[0, col_idx]

        # Plot the spike
        spike_x = x_vals_pt[idx_a]
        spike_y = q_vals_pt[idx_a]
        ax.plot([spike_x, spike_x], [0, spike_y], color=colors["intensity"], linewidth=2)
        ax.plot(spike_x, spike_y, marker="o", color=colors["intensity"])

        # Column title
        ax.set_title(f"Point Load @ {lt.capitalize()}", fontsize=12)

        # y-label only on the left
        if col_idx == 0:
            ax.set_ylabel(r"$q(x)\,[kN]$")

        # Remove grid, keep scientific notation
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.grid(False)

    # ---------------------------
    #  Row 1 => Distributed Loads
    # ---------------------------
    for col_idx, lt in enumerate(dist_load_types):
        resp_dist = roark_distributed_load_response(x_vals_dist, L, E, I, w, lt)
        q_vals_dist = convert_data("intensity", resp_dist["intensity"])  # -> [kN/m]

        ax = axes[1, col_idx]

        # Plot the distributed load with fill_between() to emphasize it
        ax.plot(x_vals_dist, q_vals_dist, color=colors["intensity"])
        ax.fill_between(x_vals_dist, q_vals_dist, 0, color=colors["intensity"], alpha=0.25)

        ax.set_title(f"{lt.capitalize()} Load", fontsize=12)

        if col_idx == 0:
            ax.set_ylabel(r"$q(x)\,[kN/m]$")

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.grid(False)

        # Optionally label x on bottom row
        ax.set_xlabel(r"$x\,(\mathrm{m})$")

    fig.tight_layout()
    # Save and show
    fig.savefig(os.path.join(SAVE_DIR, 'q_intensities_2x3.png'))
    plt.show()

# ------------------------------------------------------------------------------------
#   FIGURE 5_4 & FIGURE 6_4: 5-row layout (q, u, θ, V, M) for point / distributed
# ------------------------------------------------------------------------------------

def plot_point_load_fig5_4(L, E, I, P):
    """
    Plots 4-row layout for a beam under point loads (no q(x)):
    Rows = [u_y(x), θ_z(x), V(x), M(x)]
    Columns = [end, mid, quarter]
    """
    x_vals = np.linspace(0, L, 200)
    load_types = ["end", "mid", "quarter"]

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12), sharex=True)
    # No overall suptitle

    for col_idx, lt in enumerate(load_types):
        resp = roark_point_load_response(x_vals, L, E, I, P, lt)

        # Convert response to the final units
        u_vals = convert_data("deflection", resp["deflection"])  # mm
        th_vals = convert_data("rotation", resp["rotation"])     # degrees
        V_vals = convert_data("shear", resp["shear"])            # kN
        M_vals = convert_data("moment", resp["moment"])          # kN·m

        # -- Row 0 => u(x) --
        ax_u = axes[0, col_idx]
        ax_u.plot(x_vals, u_vals, color=colors["deflection"])
        if col_idx == 0:
            ax_u.set_ylabel(r"$u_{y}(x)\,[mm]$")
        ax_u.set_title(f"Point Load @ {lt.capitalize()}", fontsize=12)

        # -- Row 1 => θ(x) --
        ax_th = axes[1, col_idx]
        ax_th.plot(x_vals, th_vals, color=colors["rotation"])
        if col_idx == 0:
            ax_th.set_ylabel(r"$\theta_{z}(x)\,[^\circ]$")

        # -- Row 2 => V(x) --
        ax_v = axes[2, col_idx]
        ax_v.step(x_vals, V_vals, where="post", color=colors["shear"])
        ax_v.fill_between(x_vals, V_vals, 0, step="post", color=colors["shear"], alpha=0.25)
        if col_idx == 0:
            ax_v.set_ylabel(r"$V(x)\,[kN]$")

        # -- Row 3 => M(x) --
        ax_m = axes[3, col_idx]
        ax_m.plot(x_vals, M_vals, color=colors["moment"])
        ax_m.fill_between(x_vals, M_vals, 0, color=colors["moment"], alpha=0.25)
        if col_idx == 0:
            ax_m.set_ylabel(r"$M(x)\,[kN \cdot m]$")
        ax_m.set_xlabel(r"$x\,(\mathrm{m})$")

    # Format
    for row in range(4):
        for col in range(3):
            axes[row, col].grid(False)
            axes[row, col].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'point_u_theta_V_M.png'))
    plt.show()


def plot_distributed_fig6_4(L, E, I, w):
    """
    Plots 4-row layout for a beam under distributed loads (no q(x)):
    Rows = [u_y(x), θ_z(x), V(x), M(x)]
    Columns = [udl, triangular, parabolic]
    """
    x_vals = np.linspace(0, L, 750)
    load_types = ["udl", "triangular", "parabolic"]

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12), sharex=True)
    # No overall suptitle

    for col_idx, lt in enumerate(load_types):
        resp = roark_distributed_load_response(x_vals, L, E, I, w, lt)

        # Convert response
        u_vals = convert_data("deflection", resp["deflection"])  # mm
        th_vals = convert_data("rotation", resp["rotation"])     # deg
        V_vals = convert_data("shear", resp["shear"])            # kN
        M_vals = convert_data("moment", resp["moment"])          # kN·m

        # -- Row 0 => u(x) --
        ax_u = axes[0, col_idx]
        ax_u.plot(x_vals, u_vals, color=colors["deflection"])
        if col_idx == 0:
            ax_u.set_ylabel(r"$u_{y}(x)\,[mm]$")
        ax_u.set_title(f"{lt.capitalize()} Load", fontsize=12)

        # -- Row 1 => θ(x) --
        ax_th = axes[1, col_idx]
        ax_th.plot(x_vals, th_vals, color=colors["rotation"])
        if col_idx == 0:
            ax_th.set_ylabel(r"$\theta_{z}(x)\,[^\circ]$")

        # -- Row 2 => V(x) --
        ax_v = axes[2, col_idx]
        ax_v.plot(x_vals, V_vals, color=colors["shear"])
        ax_v.fill_between(x_vals, V_vals, 0, color=colors["shear"], alpha=0.25)
        if col_idx == 0:
            ax_v.set_ylabel(r"$V(x)\,[kN]$")

        # -- Row 3 => M(x) --
        ax_m = axes[3, col_idx]
        ax_m.plot(x_vals, M_vals, color=colors["moment"])
        ax_m.fill_between(x_vals, M_vals, 0, color=colors["moment"], alpha=0.25)
        if col_idx == 0:
            ax_m.set_ylabel(r"$M(x)\,[kN \cdot m]$")
        ax_m.set_xlabel(r"$x\,(\mathrm{m})$")

    # Format
    for row in range(4):
        for col in range(3):
            axes[row, col].grid(False)
            axes[row, col].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'distributed_u_theta_V_M.png'))
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
    #plot_point_load_fig1(L, E, I, P)

    # (2) Figure 2: point loads => q, V, M
    #plot_point_load_fig2(L, E, I, P)

    # (3) Figure 3: distributed => q, u, theta
    #plot_distributed_fig3(L, E, I, w)

    # (4) Figure 4: distributed => q, V, M
    #plot_distributed_fig4(L, E, I, w)

    plot_load_intensities_2x3(L, E, I, P, w)

    # (5) Figure 5: point loads => u, theta, V, M
    #plot_point_load_fig5_1(L, E, I, P)
    #plot_point_load_fig5_2(L, E, I, P)
    #plot_point_load_fig5_3(L, E, I, P)
    plot_point_load_fig5_4(L, E, I, P)

    # (6) Figure 6: distributed loads => u, theta, V, M
    #plot_distributed_fig6_1(L, E, I, w)
    #plot_distributed_fig6_2(L, E, I, w)
    #plot_distributed_fig6_3(L, E, I, w)
    plot_distributed_fig6_4(L, E, I, w)


if __name__ == "__main__":
    main()

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

def plot_load_intensities(L, E, I, P, w):
    """
    Creates a 2x3 subplot figure for load intensities:
      - Row 0 => Point loads (end, mid, quarter) in [kN]
      - Row 1 => Distributed loads (UDL, triangular, parabolic) in [kN/m]
    """
    x_vals = np.linspace(0, L, 750)  # Unified x-axis for both rows

    point_load_types = ["end", "mid", "quarter"]
    dist_load_types = ["udl", "triangular", "parabolic"]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), sharex=True)

    blue_color = "blue"  # Define a single color for all loads

    # ----------------------
    #  Row 0 => Point Loads (Blue Line + Arrowhead)
    # ----------------------
    for col_idx, lt in enumerate(point_load_types):
        ax = axes[0, col_idx]

        # Compute responses, including intensity
        resp = roark_point_load_response(x_vals, L, E, I, P, lt)

        q_vals = resp["intensity"]  # Use computed intensity from Roark functions

        # Find nonzero intensity index for plotting spike
        idx_a = np.argmax(np.abs(q_vals))  # Find where intensity is applied
        spike_x = x_vals[idx_a]
        spike_q = q_vals[idx_a]

        # Plot vertical force line in blue
        ax.plot([spike_x, spike_x], [0, spike_q], color=blue_color, linewidth=2)

        # Add arrowhead in blue
        ax.plot(spike_x, spike_q, marker="v", color=blue_color, markersize=8)

        # Column title
        ax.set_title(f"Point Load @ {lt.capitalize()}", fontsize=12)

        # y-label only on the leftmost column
        if col_idx == 0:
            ax.set_ylabel(r"$P(x)\,[kN]$")  # Corrected y-axis label for point loads

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.grid(False)

    # ---------------------------
    #  Row 1 => Distributed Loads (Same Blue Color)
    # ---------------------------
    for col_idx, lt in enumerate(dist_load_types):
        ax = axes[1, col_idx]

        # Compute distributed load intensity directly from Roark formulas
        resp = roark_distributed_load_response(x_vals, L, E, I, w, lt)

        q_vals = resp["intensity"]  # Extract computed q(x) values

        # Plot the distributed load using the same blue color
        ax.plot(x_vals, q_vals, color=blue_color, linewidth=2)
        ax.fill_between(x_vals, q_vals, 0, color=blue_color, alpha=0.25)

        ax.set_title(f"{lt.capitalize()} Load", fontsize=12)

        if col_idx == 0:
            ax.set_ylabel(r"$q(x)\,[kN/m]$")  # Correct label for distributed loads

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.grid(False)

        ax.set_xlabel(r"$x\,(\mathrm{m})$")  # X-axis label on bottom row

    # Ensure shared x-axis formatting is applied
    plt.setp(axes[-1, :], xlabel=r"$x\,[\mathrm{m}]$")

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'q_intensities_2x3.png'))
    plt.show()

# ------------------------------------------------------------------------------------
#   FIGURE 5_4 & FIGURE 6_4: 5-row layout (q, u, θ, V, M) for point / distributed
# ------------------------------------------------------------------------------------

def plot_point_load(L, E, I, P):
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
        ax_m.set_xlabel(r"$x\,[\mathrm{m}]$")

    # Format
    for row in range(4):
        for col in range(3):
            axes[row, col].grid(False)
            axes[row, col].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'point_u_theta_V_M.png'))
    plt.show()


def plot_distributed(L, E, I, w):
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
        ax_m.set_xlabel(r"$x\,[\mathrm{m}]$")

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
    L = 2             # [m]
    E = 2.0e11        # [Pa]
    I = 1.002e+0      # [m^4]

    # Point load [N]
    P = 100000.0

    # Distributed load [N/m]
    w = 100000.0

    plot_load_intensities(L, E, I, P, w)

    plot_point_load(L, E, I, P)

    plot_distributed(L, E, I, w)


if __name__ == "__main__":
    main()

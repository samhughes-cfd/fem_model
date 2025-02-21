import numpy as np
import matplotlib.pyplot as plt

def cantilever_deflection_point(x, L, E, I, P, load_type):
    """Computes deflection u_y for a cantilever beam under point loads."""
    if load_type == "end":
        return - (P * x**2) / (6 * E * I) * (3*L - x)
    elif load_type == "mid":
        return - (P * x**2) / (48 * E * I) * (3*L**2 - 4*L*x + x**2) * (x >= L/2)
    elif load_type == "quarter":
        return - (P * x**2) / (192 * E * I) * (8*L**2 - 12*L*x + 3*x**2) * (x >= L/4)
    else:
        raise ValueError("Invalid load type")

def cantilever_rotation_point(x, L, E, I, P, load_type):
    """Computes rotation theta_z for a cantilever beam under point loads."""
    if load_type == "end":
        return - (P * x) / (2 * E * I) * (2*L - x)
    elif load_type == "mid":
        return - (P * x) / (16 * E * I) * (4*L - x) * (x >= L/2)
    elif load_type == "quarter":
        return - (P * x) / (64 * E * I) * (8*L - 3*x) * (x >= L/4)
    else:
        raise ValueError("Invalid load type")

def shear_force_point(x, L, P, load_type):
    """Computes shear force S(x) for point loads."""
    if load_type == "end":
        return -P * (x >= 0)
    elif load_type == "mid":
        return -P * (x >= L/2)
    elif load_type == "quarter":
        return -P * (x >= L/4)
    else:
        raise ValueError("Invalid load type")

def bending_moment_point(x, L, P, load_type):
    """Computes bending moment M(x) for point loads."""
    if load_type == "end":
        return -P * (L - x) * (x >= 0)
    elif load_type == "mid":
        return -P * (L/2 - x) * (x >= L/2)
    elif load_type == "quarter":
        return -P * (L/4 - x) * (x >= L/4)
    else:
        raise ValueError("Invalid load type")

# Define parameters
L = 8.0        # Length of the beam (m)
E = 2e11       # Young's modulus (Pa) for steel
I = 2.67e-7    # Moment of inertia (m^4)
P = 1000       # Point load (N)

# Point load cases (End, Mid-span, Quarter-span)
point_load_cases = ["end", "mid", "quarter"]

# Discretize beam length
x_vals = np.linspace(0, L, 100)

# Compute results for each point load case
deflections = {case: cantilever_deflection_point(x_vals, L, E, I, P, case) for case in point_load_cases}
rotations = {case: cantilever_rotation_point(x_vals, L, E, I, P, case) for case in point_load_cases}
shear_forces = {case: shear_force_point(x_vals, L, P, case) for case in point_load_cases}
bending_moments = {case: bending_moment_point(x_vals, L, P, case) for case in point_load_cases}

# Explicit formulas for legend labels
formulas = {
    "deflection": {
        "end": r"$u_y (x) = -\frac{P x^2}{6 E I} (3L - x)$",
        "mid": r"$u_y (x) = -\frac{P x^2}{48 E I} (3L^2 - 4Lx + x^2)$",
        "quarter": r"$u_y (x) = -\frac{P x^2}{192 E I} (8L^2 - 12Lx + 3x^2)$"
    },
    "rotation": {
        "end": r"$\theta_z (x) = -\frac{P x}{2 E I} (2L - x)$",
        "mid": r"$\theta_z (x) = -\frac{P x}{16 E I} (4L - x)$",
        "quarter": r"$\theta_z (x) = -\frac{P x}{64 E I} (8L - 3x)$"
    },
    "shear": {
        "end": r"$S (x) = -P$",
        "mid": r"$S (x) = -P$ (x ≥ L/2)",
        "quarter": r"$S (x) = -P$ (x ≥ L/4)"
    },
    "moment": {
        "end": r"$M (x) = -P (L - x)$",
        "mid": r"$M (x) = -P (L/2 - x)$ (x ≥ L/2)",
        "quarter": r"$M (x) = -P (L/4 - x)$ (x ≥ L/4)"
    }
}

# Titles for each column (load case)
column_titles = {
    "end": "Point Load at Free End",
    "mid": "Point Load at Mid-Span",
    "quarter": "Point Load at Quarter-Span"
}

# Y-axis labels for each row (quantity)
row_labels = {
    "deflection": "Deflection (mm)",
    "rotation": "Rotation (°)",
    "shear": "Shear Force (kN)",
    "moment": "Bending Moment (kNm)"
}

# Custom colors
colors = {
    "deflection": "#4F81BD",  # Blue
    "rotation": "#4F81BD",    # Blue
    "shear": "#9BBB59",      # Green
    "moment": "#C0504D"      # Red
}

# Create subplots
fig, axes = plt.subplots(4, 3, figsize=(15, 12))

# Explicit order for plotting: End → Mid → Quarter
for i, case in enumerate(["end", "mid", "quarter"]):
    for row, (category, data) in enumerate(zip(
            ["deflection", "rotation", "shear", "moment"],
            [deflections, rotations, shear_forces, bending_moments])):

        # Apply unit conversions
        converted_data = data[case] * (1000 if category == "deflection" else np.degrees(1) if category == "rotation" else 1/1000)

        # Ensure discontinuity in shear force by manually introducing breakpoints
        if category == "shear":
            x_steps = np.insert(x_vals, np.searchsorted(x_vals, [L/4, L/2]), [L/4, L/2])  # Add breakpoints at quarter-span and mid-span
            shear_steps = np.insert(converted_data, np.searchsorted(x_vals, [L/4, L/2]), np.nan)  # Insert NaNs to break the line
            axes[row, i].step(x_steps, shear_steps, color=colors[category], where="mid", label=formulas[category][case])

        else:
            axes[row, i].plot(x_vals, converted_data, color=colors[category], label=formulas[category][case])

        # Set titles explicitly for each column
        if row == 0:
            axes[row, i].set_title(column_titles[case], fontsize=12)

        # Set y-axis labels explicitly for each row
        if i == 0:
            axes[row, i].set_ylabel(row_labels[category], fontsize=12)

        # Set x-axis label only on the bottom row
        if row == 3:
            axes[row, i].set_xlabel("Position along beam (m)")

        # Add legend without a surrounding box
        axes[row, i].legend(frameon=False)

        # Format y-axis ticks in scientific notation
        axes[row, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

plt.tight_layout()
plt.show()
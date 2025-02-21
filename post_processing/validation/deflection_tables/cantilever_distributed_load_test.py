import numpy as np
import matplotlib.pyplot as plt

def cantilever_deflection(x, L, E, I, w, load_type):
    """Computes deflection u_y for a cantilever beam under different load types."""
    if load_type == "udl":
        return -(w * x**2) / (24 * E * I) * (L**3 - 2*L*x + x**2)
    elif load_type == "triangular":
        return - (w * x**2) / (120 * L * E * I) * (10*L**3 - 10*L**2*x + 5*L*x**2 - x**3)
    elif load_type == "parabolic":
        return -(w * x**2) / (60 * L**3 * E * I) * (6*L**5 - 10*L**4*x + 5*L**3*x**2 - x**5)
    else:
        raise ValueError("Invalid load type")

def cantilever_rotation(x, L, E, I, w, load_type):
    """Computes rotation theta_z for a cantilever beam under different load types."""
    if load_type == "udl":
        return -(w * x) / (6 * E * I) * (L**3 - 3*L*x + x**2)
    elif load_type == "triangular":
        return - (w * x) / (24 * L * E * I) * (4*L**3 - 6*L**2*x + 4*L*x**2 - x**3)
    elif load_type == "parabolic":
        return -(w * x) / (20 * L**3 * E * I) * (5*L**5 - 10*L**4*x + 6*L**3*x**2 - x**5)
    else:
        raise ValueError("Invalid load type")

def shear_force(x, L, w, load_type):
    """Computes shear force S(x) for different load types."""
    if load_type == "udl":
        return -w * (L - x)
    elif load_type == "triangular":
        return - (w * x**2) / (2 * L)
    elif load_type == "parabolic":
        return - (w * x**3) / (3 * L**2)
    else:
        raise ValueError("Invalid load type")

def bending_moment(x, L, w, load_type):
    """Computes bending moment M(x) for different load types."""
    if load_type == "udl":
        return -w * (L*x - x**2/2)
    elif load_type == "triangular":
        return - (w * x**3) / (6 * L)
    elif load_type == "parabolic":
        return - (w * x**4) / (12 * L**2)
    else:
        raise ValueError("Invalid load type")

# Define parameters
L = 8.0        # Length of the beam (m)
E = 2e11       # Young's modulus (Pa) for steel
I = 2.67e-7    # Moment of inertia (m^4)

# Load intensities (always in order: UDL → Triangular → Parabolic)
loads = {"udl": 1000, "triangular": 1000, "parabolic": 1000}

# Discretize beam length
x_vals = np.linspace(0, L, 100)

# Compute results using the functions
deflections = {lt: cantilever_deflection(x_vals, L, E, I, loads[lt], lt) for lt in loads}
rotations = {lt: cantilever_rotation(x_vals, L, E, I, loads[lt], lt) for lt in loads}
shear_forces = {lt: shear_force(x_vals, L, loads[lt], lt) for lt in loads}
bending_moments = {lt: bending_moment(x_vals, L, loads[lt], lt) for lt in loads}

# Explicit formulas for legend labels
formulas = {
    "deflection": {
        "udl": r"$u_y (x) = -\frac{w x^2}{24 E I}(L^3-2Lx+x^2)$",
        "triangular": r"$u_y (x) = -\frac{w x^2}{120 L E I}(10L^3-10L^2x+5Lx^2-x^3)$",
        "parabolic": r"$u_y (x) = -\frac{w x^2}{60 L^3 E I}(6L^5-10L^4x+5L^3x^2-x^5)$"
    },
    "rotation": {
        "udl": r"$\theta_z (x) = -\frac{w x}{6 E I}(L^3-3Lx+x^2)$",
        "triangular": r"$\theta_z (x) = -\frac{w x}{24 L E I}(4L^3-6L^2x+4Lx^2-x^3)$",
        "parabolic": r"$\theta_z (x) = -\frac{w x}{20 L^3 E I}(5L^5-10L^4x+6L^3x^2-x^5)$"
    },
    "shear": {
        "udl": r"$S (x) = -w (L - x)$",
        "triangular": r"$S (x) = -\frac{w x^2}{2 L}$",
        "parabolic": r"$S (x) = -\frac{w x^3}{3 L^2}$"
    },
    "moment": {
        "udl": r"$M (x) = -w (Lx - \frac{x^2}{2})$",
        "triangular": r"$M (x) = -\frac{w x^3}{6 L}$",
        "parabolic": r"$M (x) = -\frac{w x^4}{12 L^2}$"
    }
}

# Custom colors
colors = {
    "deflection": "#4F81BD",
    "rotation": "#4F81BD",
    "shear": "#9BBB59",
    "moment": "#C0504D"
}

# Create subplots
fig, axes = plt.subplots(4, 3, figsize=(15, 12))

# Explicit order for plotting: UDL → Triangular → Parabolic
for i, lt in enumerate(["udl", "triangular", "parabolic"]):
    for row, (category, data) in enumerate(zip(
            ["deflection", "rotation", "shear", "moment"],
            [deflections, rotations, shear_forces, bending_moments])):

        # Apply unit conversions
        converted_data = data[lt] * (1000 if category == "deflection" else np.degrees(1) if category == "rotation" else 1/1000)

        # Plot the converted data with assigned color
        axes[row, i].plot(x_vals, converted_data, color=colors[category], label=formulas[category][lt])

        # Set titles explicitly for each column
        if row == 0:
            axes[row, i].set_title(f"{lt.capitalize()} Load", fontsize=12)

        # Set y-axis labels explicitly for each row
        if i == 0:
            axes[row, i].set_ylabel(f"{category.capitalize()} ({'mm' if category == 'deflection' else '°' if category == 'rotation' else 'kN' if category == 'shear' else 'kNm'})", fontsize=12)

        # Set x-axis label only on the bottom row
        if row == 3:
            axes[row, i].set_xlabel("Position along beam (m)")

        # Add legend without a surrounding box
        axes[row, i].legend(frameon=False)

        # Format y-axis ticks in scientific notation
        axes[row, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# ==============================
#    BEAM RESPONSE FUNCTIONS
# ==============================

def compute_deflection(x, L, E, I, P, load_type):
    """
    Computes transverse deflection \( u_y(x) \) of a cantilever beam under a point load \( P \),
    ensuring \( C^1 \) continuity at the load location.

    Governing equation:
        \( EI \frac{d^2 u_y}{dx^2} = M(x) \), with piecewise bending moment \( M(x) \).

    Boundary conditions:
        - \( u_y(0) = 0 \) (fixed end, zero displacement)
        - \( \frac{du_y}{dx} \bigg|_{x=0} = 0 \) (zero rotation at fixed end)
        - \( C^1 \) continuity at \( x = a \) (deflection and slope must match)

    Parameters:
        x (array): Discretized beam coordinate values.
        L (float): Beam length.
        E (float): Young’s modulus.
        I (float): Second moment of area.
        P (float): Point load (applied downward).
        load_type (str): Load position ("end", "mid", "quarter").

    Returns:
        array: Deflection \( u_y(x) \), ensuring smooth transitions at \( x = a \).
    """
    u_y = np.zeros_like(x)

    if load_type == "end":
        return - (P * x**2) / (6 * E * I) * (3 * L - x)

    a = L / 2 if load_type == "mid" else L / 4

    def u1(xv):
        return -(P / (E * I)) * (a * (xv**2) / 2 - (xv**3) / 6)

    def du1dx(xv):
        return -(P / (E * I)) * (a * xv - (xv**2) / 2)

    u1_a, du1_a = u1(a), du1dx(a)

    C3, C4 = du1_a, u1_a - du1_a * a

    def u2(xv):
        return C3 * xv + C4

    for i, xv in enumerate(x):
        u_y[i] = u1(xv) if xv < a else u2(xv)

    return u_y


def compute_rotation(x, L, E, I, P, load_type):
    """
    Computes slope \( \theta_z(x) = \frac{du_y}{dx} \) of a cantilever beam under a point load \( P \),
    ensuring \( C^1 \) continuity.

    Governing equation:
        \( EI \frac{d \theta_z}{dx} = M(x) \).
    """
    theta_z = np.zeros_like(x)

    if load_type == "end":
        return - (P * x) / (2 * E * I) * (2 * L - x)

    a = L / 2 if load_type == "mid" else L / 4

    def du1dx(xv):
        return -(P / (E * I)) * (a * xv - (xv**2) / 2)

    slope_a = du1dx(a)

    for i, xv in enumerate(x):
        theta_z[i] = du1dx(xv) if xv < a else slope_a

    return theta_z


def compute_shear_force(x, L, P, load_type):
    """
    Computes internal shear force \( V(x) \) for a cantilever beam under a point load \( P \).
    Governing equation:
        \( V(x) = \frac{dM}{dx} \).
    """
    V = np.zeros_like(x)
    if load_type == "end":
        return -P * np.ones_like(x)

    a = L / 2 if load_type == "mid" else L / 4
    for i, xv in enumerate(x):
        V[i] = -P if xv < a else 0

    return V


def compute_bending_moment(x, L, P, load_type):
    """
    Computes internal bending moment \( M(x) \) for a cantilever beam under a point load \( P \).
    Governing equation:
        \( M(x) = \int V(x) \, dx \).
    """
    M = np.zeros_like(x)
    if load_type == "end":
        return -P * (L - x)

    a = L / 2 if load_type == "mid" else L / 4
    for i, xv in enumerate(x):
        M[i] = -P * (a - xv) if xv < a else 0

    return M


def compute_beam_response(x, L, E, I, P, load_type):
    """Computes all beam responses: deflection, rotation, shear force, and bending moment."""
    return {
        "deflection": compute_deflection(x, L, E, I, P, load_type),
        "rotation": compute_rotation(x, L, E, I, P, load_type),
        "shear": compute_shear_force(x, L, P, load_type),
        "moment": compute_bending_moment(x, L, P, load_type),
    }


# ==============================
#        PLOTTING FUNCTION
# ==============================

def plot_beam_responses(x_vals, responses, load_cases):
    """Plots deflection, rotation, shear force, and bending moment for different load cases."""
    column_titles = {
        "end": "Point Load at Free End",
        "mid": "Point Load at Mid-Span",
        "quarter": "Point Load at Quarter-Span"
    }
    
    row_labels = ["Deflection (mm)", "Rotation (°)", "Shear Force (kN)", "Bending Moment (kNm)"]
    colors = ["#4F81BD", "#4F81BD", "#9BBB59", "#C0504D"]

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))

    for i, case in enumerate(load_cases):
        row_data = [responses[case]["deflection"] * 1000, 
                    np.degrees(responses[case]["rotation"]), 
                    responses[case]["shear"] / 1000, 
                    responses[case]["moment"] / 1000]

        for row, (data, ylabel, color) in enumerate(zip(row_data, row_labels, colors)):
            ax = axes[row, i]

            # Shear force should be step plot
            if row == 2:
                ax.step(x_vals, data, where="post", color=color)
            else:
                ax.plot(x_vals, data, color=color)

            ax.set_title(column_titles[case] if row == 0 else "")
            ax.set_ylabel(ylabel if i == 0 else "")
            ax.set_xlabel("Position x (m)" if row == 3 else "")

            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plt.show()


# ==============================
#        MAIN FUNCTION
# ==============================

def main():
    """Main execution function to compute and plot beam responses."""
    L, E, I, P = 8.0, 2.0e11, 2.67e-7, 1000
    load_cases = ["end", "mid", "quarter"]
    x_vals = np.linspace(0, L, 200)
    responses = {case: compute_beam_response(x_vals, L, E, I, P, case) for case in load_cases}
    plot_beam_responses(x_vals, responses, load_cases)


if __name__ == "__main__":
    main()
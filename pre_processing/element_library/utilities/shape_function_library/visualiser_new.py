import numpy as np
import matplotlib.pyplot as plt
import logging
from labellines import labelLine, labelLines

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def shape_functions(xi: np.ndarray, L: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute shape functions and derivatives for a 3D Euler-Bernoulli beam.

    Args:
        xi: Natural coordinates in range [-1, 1]
        L: Beam element length (default = 1.0)

    Returns:
        - N_matrix (g, 12, 6): Shape functions for translation and rotation DOFs
        - dN_dxi_matrix (g, 12, 6): First derivatives
        - d2N_dxi2_matrix (g, 12, 6): Second derivatives
    """
    xi = np.atleast_1d(xi)
    g = xi.shape[0]  # Number of Gauss points

    # **1. Axial Shape Functions (Linear Lagrange)**
    N1 = 0.5 * (1 - xi)
    N7 = 0.5 * (1 + xi)
    dN1_dxi = -0.5 * np.ones(g)
    dN7_dxi = 0.5 * np.ones(g)
    d2N1_dxi2 = np.zeros(g)
    d2N7_dxi2 = np.zeros(g)

    # **2. Bending in XY Plane (Hermite Cubic)**
    N2 = 0.5 * (1 - 3 * xi**2 + 2 * xi**3)
    N8 = 0.5 * (3 * xi**2 - 2 * xi**3 + 1)
    N3 = 0.5 * L * (xi - 2 * xi**2 + xi**3)
    N9 = 0.5 * L * (-xi**2 + xi**3)

    # **Derivatives**
    dN2_dxi = -3 * xi + 2 * xi**2
    dN8_dxi = 3 * xi - 2 * xi**2
    dN3_dxi = 0.5 * L * (1 - 4 * xi + 3 * xi**2)
    dN9_dxi = 0.5 * L * (-2 * xi + 3 * xi**2)

    d2N2_dxi2 = -3 + 4 * xi
    d2N8_dxi2 = 3 - 4 * xi
    d2N3_dxi2 = 0.5 * L * (-4 + 6 * xi)
    d2N9_dxi2 = 0.5 * L * (-2 + 6 * xi)

    # **3. Bending in XZ Plane (Reuse Hermite Cubic Functions)**
    N4, N10 = N2, N8
    N5, N11 = N3, N9
    dN4_dxi, dN10_dxi = dN2_dxi, dN8_dxi
    dN5_dxi, dN11_dxi = dN3_dxi, dN9_dxi
    d2N4_dxi2, d2N10_dxi2 = d2N2_dxi2, d2N8_dxi2
    d2N5_dxi2, d2N11_dxi2 = d2N3_dxi2, d2N9_dxi2

    # **4. Torsion Shape Functions (Linear Interpolation)**
    N6, N12 = N1, N7
    dN6_dxi, dN12_dxi = dN1_dxi, dN7_dxi
    d2N6_dxi2, d2N12_dxi2 = d2N1_dxi2, d2N7_dxi2

    # **5. Assemble Shape Function Matrices**
    N_matrix = np.zeros((g, 12, 6))
    dN_dxi_matrix = np.zeros((g, 12, 6))
    d2N_dxi2_matrix = np.zeros((g, 12, 6))

    # Assign values to matrices
    N_matrix[:, 0, 0] = N1   # Axial displacement (u_x)
    N_matrix[:, 6, 0] = N7
    N_matrix[:, 1, 1] = N2   # Transverse displacement (u_y)
    N_matrix[:, 7, 1] = N8
    N_matrix[:, 2, 2] = N4   # Transverse displacement (u_z)
    N_matrix[:, 8, 2] = N10
    N_matrix[:, 3, 3] = N6   # Torsion (θ_x)
    N_matrix[:, 9, 3] = N12
    N_matrix[:, 4, 4] = N5   # Bending rotation (θ_y)
    N_matrix[:, 10, 4] = N11
    N_matrix[:, 5, 5] = N3   # Bending rotation (θ_z)
    N_matrix[:, 11, 5] = N9

    dN_dxi_matrix[:, 0, 0] = dN1_dxi
    dN_dxi_matrix[:, 6, 0] = dN7_dxi
    dN_dxi_matrix[:, 1, 1] = dN2_dxi
    dN_dxi_matrix[:, 7, 1] = dN8_dxi
    dN_dxi_matrix[:, 2, 2] = dN4_dxi
    dN_dxi_matrix[:, 8, 2] = dN10_dxi
    dN_dxi_matrix[:, 3, 3] = dN6_dxi
    dN_dxi_matrix[:, 9, 3] = dN12_dxi
    dN_dxi_matrix[:, 4, 4] = dN5_dxi
    dN_dxi_matrix[:, 10, 4] = dN11_dxi
    dN_dxi_matrix[:, 5, 5] = dN3_dxi
    dN_dxi_matrix[:, 11, 5] = dN9_dxi

    d2N_dxi2_matrix[:, 1, 1] = d2N2_dxi2
    d2N_dxi2_matrix[:, 7, 1] = d2N8_dxi2
    d2N_dxi2_matrix[:, 2, 2] = d2N4_dxi2
    d2N_dxi2_matrix[:, 8, 2] = d2N10_dxi2
    d2N_dxi2_matrix[:, 5, 5] = d2N3_dxi2
    d2N_dxi2_matrix[:, 11, 5] = d2N9_dxi2

    return N_matrix, dN_dxi_matrix, d2N_dxi2_matrix

def plot_shape_functions():
    """Plots shape functions in two separate 2x2 grids: one for Node 1, one for Node 2."""
    xi_values = np.linspace(-1, 1, 100)
    N, dN_dxi, d2N_dxi2 = shape_functions(xi_values)

    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    fig1.suptitle("Euler-Bernoulli Beam Shape Functions (Node 1)", fontsize=14)

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    fig2.suptitle("Euler-Bernoulli Beam Shape Functions (Node 2)", fontsize=14)

    # **Define subplot titles and corresponding DOF indices**
    groups = {
        "Axial Displacement & Strain ($u_x, \\varepsilon_x$)": [0, 6],  # N1, N7
        "Bending/Rotation in XY Plane ($u_z, \\theta_y, \\kappa_y$)": [2, 4, 8, 10],  # N2, N8, N3, N9
        "Bending/Rotation in XZ Plane ($u_y, \\theta_z, \\kappa_z$)": [1, 5, 7, 11],  # N4, N10, N5, N11
        "Torsion & Torsional Strain ($\\theta_x, \\gamma_x$)": [3, 9]  # N6, N12
    }

    for (ax1, ax2), (title, dof_indices) in zip(zip(axes1.flatten(), axes2.flatten()), groups.items()):
        for idx in dof_indices:
            node_number = 1 if idx < 6 else 2  # Map DOF index to Node 1 or Node 2
            dof_local = idx % 6  # Local DOF within each node

            # **Select the correct subplot**
            ax = ax1 if node_number == 1 else ax2

            # **Plot shape functions and derivatives**
            line1, = ax.plot(xi_values, N[:, idx, dof_local], linestyle='-', label=r"$N_{{{}}}(\xi)$".format(dof_local+1))
            line2, = ax.plot(xi_values, dN_dxi[:, idx, dof_local], linestyle='--', label=r"$\frac{{dN_{{{}}}}}{{d\xi}}$".format(dof_local+1))
            line3, = ax.plot(xi_values, d2N_dxi2[:, idx, dof_local], linestyle=':', label=r"$\frac{{d^2N_{{{}}}}}{{d\xi^2}}$".format(dof_local+1))

            # **Embed labels into the lines**
            labelLines([line1, line2, line3], align=True, fontsize=10)

        ax1.set_title(title)
        ax1.set_xlabel(r"Natural Coordinate $\xi$")
        ax1.set_ylabel("Shape Function Values")
        ax1.grid(True)

        ax2.set_title(title)
        ax2.set_xlabel(r"Natural Coordinate $\xi$")
        ax2.set_ylabel("Shape Function Values")
        ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_shape_functions()
import numpy as np
import matplotlib.pyplot as plt
import logging
from labellines import labelLines

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

cases = {
    # Axial Deformation and Torsion
    "Axial Deformation and Torsion": {
        "axial_dofs": [0, 6],  # N1, N7 (u_x)
        "torsional_dofs": [5, 11],  # N6, N12 (θ_x)
        "title": "Axial Deformation and Torsion",
        "axial_labels": ["$N_1(\\xi)$", "$N_7(\\xi)$"],
        "torsional_labels": ["$N_6(\\xi)$", "$N_{12}(\\xi)$"]
    },
    
    # XY Bending - Transverse and Rotational
    "Bending in XY Plane": {
        "transverse_dofs": [1, 7],  # N2, N8 (u_z)
        "rotational_dofs": [2, 8],  # N3, N9 (θ_y)
        "title": "Bending in XY Plane",
        "transverse_labels": ["$N_2(\\xi)$", "$N_8(\\xi)$"],
        "rotational_labels": ["$N_3(\\xi)$", "$N_9(\\xi)$"]
    },
    
    # XZ Bending - Transverse and Rotational
    "Bending in XZ Plane": {
        "transverse_dofs": [3, 9],  # N4, N10 (u_y)
        "rotational_dofs": [4, 10],  # N5, N11 (θ_z)
        "title": "Bending in XZ Plane",
        "transverse_labels": ["$N_4(\\xi)$", "$N_{10}(\\xi)$"],
        "rotational_labels": ["$N_5(\\xi)$", "$N_{11}(\\xi)$"]
    }
}

def plot_shape_functions():
    """Plots shape functions with 2x2 subplots, node markers, and non-active nodes at 25% opacity."""
    xi_values = np.linspace(-1, 1, 100)
    N, dN_dxi, d2N_dxi2 = shape_functions(xi_values)

    for case_name, case_data in cases.items():
        # Create 2x2 subplot grid for all cases
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
        fig.suptitle(case_data["title"], fontsize=14)

        # Plot axial deformation or transverse displacement (first row)
        for ax, node_number in zip(axes[0], [1, 2]):
            if "Axial" in case_name:
                idx = case_data["axial_dofs"][0 if node_number == 1 else 1]
                label = case_data["axial_labels"][0 if node_number == 1 else 1]
            else:
                idx = case_data["transverse_dofs"][0 if node_number == 1 else 1]
                label = case_data["transverse_labels"][0 if node_number == 1 else 1]

            dof_local = idx % 6
            function_name = label.split("(")[0][1:]  # Extract N_X from label

            # Plot shape functions and derivatives
            line1, = ax.plot(xi_values, N[:, idx, dof_local], 
                           linestyle="-", label=label)
            line2, = ax.plot(xi_values, dN_dxi[:, idx, dof_local],
                           linestyle="--", label=rf"$\frac{{d{function_name}}}{{d\xi}}$")
            line3, = ax.plot(xi_values, d2N_dxi2[:, idx, dof_local],
                           linestyle=":", label=rf"$\frac{{d^2{function_name}}}{{d\xi^2}}$")

            # Add labels directly to the lines
            labelLines([line1, line2, line3], align=True, fontsize=10)

            # Add node markers
            for marker_node in [1, 2]:
                node_x = -1 if marker_node == 1 else 1
                node_y = 0  # Fixed y-coordinate for node markers
                opacity = 1.0 if marker_node == node_number else 0.25  # 25% opacity for non-active node
                ax.plot(node_x, node_y, 'ko', markersize=8, alpha=opacity)
                #ax.text(node_x, node_y, f"Node {marker_node}", fontsize=12, 
                        #ha='right' if marker_node == 1 else 'left', va='bottom', alpha=opacity)

            # Set subplot title and labels
            ax.set_title(f"Node {node_number} (DOF {idx})")
            ax.set_xlabel(r"$\xi$ [-]")
            ax.grid(True)

            # Add y-axis label only for Node 1 subplot
            if node_number == 1:
                ax.set_ylabel(r"$f(\xi)$ [-]")

        # Plot torsional or rotational deformation (second row)
        for ax, node_number in zip(axes[1], [1, 2]):
            if "Axial" in case_name:
                idx = case_data["torsional_dofs"][0 if node_number == 1 else 1]
                label = case_data["torsional_labels"][0 if node_number == 1 else 1]
            else:
                idx = case_data["rotational_dofs"][0 if node_number == 1 else 1]
                label = case_data["rotational_labels"][0 if node_number == 1 else 1]

            dof_local = idx % 6
            function_name = label.split("(")[0][1:]  # Extract N_X from label

            # Plot shape functions and derivatives
            line1, = ax.plot(xi_values, N[:, idx, dof_local], 
                           linestyle="-", label=label)
            line2, = ax.plot(xi_values, dN_dxi[:, idx, dof_local],
                           linestyle="-", label=rf"$\frac{{d{function_name}}}{{d\xi}}$")
            line3, = ax.plot(xi_values, d2N_dxi2[:, idx, dof_local],
                           linestyle="-", label=rf"$\frac{{d^2{function_name}}}{{d\xi^2}}$")

            # Add labels directly to the lines
            labelLines([line1, line2, line3], align=True, fontsize=10)

            # Add node markers
            for marker_node in [1, 2]:
                node_x = -1 if marker_node == 1 else 1
                node_y = 0  # Fixed y-coordinate for node markers
                opacity = 1.0 if marker_node == node_number else 0.25  # 25% opacity for non-active node
                ax.plot(node_x, node_y, 'ko', markersize=8, alpha=opacity)
                #ax.text(node_x, node_y, f"Node {marker_node}", fontsize=12, 
                        #ha='right' if marker_node == 1 else 'left', va='bottom', alpha=opacity)

            # Set subplot title and labels
            ax.set_title(f"Node {node_number} (DOF {idx})")
            ax.set_xlabel(r"$\xi$ [-]")
            ax.grid(True)

            # Add y-axis label only for Node 1 subplot
            if node_number == 1:
                ax.set_ylabel(r"$f(\xi)$ [-]")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plot_shape_functions()
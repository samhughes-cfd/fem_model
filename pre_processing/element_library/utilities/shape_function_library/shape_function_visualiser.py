import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from labellines import labelLines
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the project root directory is added to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

# Debug sys.path
logging.debug("Available module paths:")
for path in sys.path:
    logging.debug(path)

# Import shape functions with error handling
try:
    from pre_processing.element_library.utilities.shape_function_library.euler_bernoulli_sf import euler_bernoulli_shape_functions
    from pre_processing.element_library.utilities.shape_function_library.timoshenko_sf import timoshenko_shape_functions
except ModuleNotFoundError as e:
    logging.error(f"Module import error: {e}")
    logging.error("Check if pre_processing is correctly structured and in sys.path")
    sys.exit(1)
except ImportError as e:
    logging.error(f"Import error: {e}")
    logging.error("Possible circular import or missing function in module.")
    sys.exit(1)

def plot_shape_functions():
    """
    Plots the shape functions (N), their first derivatives (dN/dxi), and second derivatives (d²N/dxi²)
    for both Euler-Bernoulli and Timoshenko beam elements, using inline labels.
    """
    try:
        xi_values = np.linspace(-1, 1, 100)  # Natural coordinate range
        L = 1.0  # Example element length

        # Compute Shape Functions
        N_eb, dN_eb, d2N_eb = zip(*[euler_bernoulli_shape_functions(xi, L, poly_order=3) for xi in xi_values])
        N_t, dN_t, d2N_t = zip(*[timoshenko_shape_functions(xi, L, poly_order=2) for xi in xi_values])

        # Convert to numpy arrays
        N_eb, dN_eb, d2N_eb = np.array(N_eb), np.array(dN_eb), np.array(d2N_eb)
        N_t, dN_t, d2N_t = np.array(N_t), np.array(dN_t), np.array(d2N_t)

        # Labels for Degrees of Freedom (DOFs)
        labels_1 = [r"$N_1(\xi)$", r"$N_2(\xi)$", r"$N_3(\xi)$", r"$N_4(\xi)$", r"$N_5(\xi)$", r"$N_6(\xi)$"]
        labels_2 = [r"$\frac{dN_1(\xi)}{d\xi}$", r"$\frac{dN_2(\xi)}{d\xi}$", r"$\frac{dN_3(\xi)}{d\xi}$", r"$\frac{dN_4(\xi)}{d\xi}$", r"$\frac{dN_5(\xi)}{d\xi}$", r"$\frac{dN_6(\xi)}{d\xi}$"]
        labels_3 = [r"$\frac{d^2N_1(\xi)}{d\xi^2}$", r"$\frac{d^2N_2(\xi)}{d\xi^2}$", r"$\frac{d^2N_3(\xi)}{d\xi^2}$", r"$\frac{d^2N_4(\xi)}{d\xi^2}$", r"$\frac{d^2N_5(\xi)}{d\xi^2}$", r"$\frac{d^2N_6(\xi)}{d\xi^2}$"]

        # Function for plotting
        def plot_functions(title, N, dN, d2N):
            fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
            fig.suptitle(title, fontsize=14)

            for i in range(N.shape[1]):
                axes[0].plot(xi_values, N[:, i], label=labels_1[i])
                axes[1].plot(xi_values, dN[:, i], label=labels_2[i])
                axes[2].plot(xi_values, d2N[:, i], label=labels_3[i])

            axes[0].set_ylabel(r"$N_i(\xi)$ Shape Functions")
            axes[1].set_ylabel(r"$\frac{dN_i(\xi)}{d\xi}$ 1st Derivatives")
            axes[2].set_ylabel(r"$\frac{d^2N_i(\xi)}{d\xi^2}$ 2nd Derivatives")
            axes[2].set_xlabel(r"$\xi$")
            
            for ax in axes:
                labelLines(ax.get_lines(), zorder=2.5)
            
            plt.show()

        # Plot for Euler-Bernoulli
        plot_functions("Euler-Bernoulli Beam Shape Functions", N_eb, dN_eb, d2N_eb)

        # Plot for Timoshenko
        plot_functions("Timoshenko Beam Shape Functions", N_t, dN_t, d2N_t)
    except Exception as e:
        logging.error(f"Error during shape function computation or plotting: {e}")

# Run the visualization
if __name__ == "__main__":
    plot_shape_functions()
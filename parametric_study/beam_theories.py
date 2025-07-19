import numpy as np
import matplotlib.pyplot as plt

# Beam and material properties (S275JR CoTide C-channel)
E = 2.1e11       # Young's modulus [Pa]
G = 8.1e10       # Shear modulus [Pa]
A = 0.00131      # Cross-sectional area [m^2]
I = 2.08769e-06  # Second moment of area [m^4]
kappa = 5/6      # Shear correction factor
L = 2.0          # Beam length [m]
P = -500.0       # Point load [N] downward

# Discretisation
x = np.linspace(0, L, 200)

# Euler-Bernoulli theory
w_eb = (P / (6 * E * I)) * (3 * L * x**2 - x**3)
theta_eb = (P / (2 * E * I)) * (3 * L * x - x**2)

# Timoshenko theory
w_tim = w_eb + (P * x) / (kappa * A * G)
theta_tim = theta_eb

# Levinson theory
w_lev = w_eb + (P * x / (G * A)) * (1 - (x**2 / L**2))
theta_shear_lev = (P / (G * A)) * (1 - 3 * x**2 / L**2)
theta_lev = theta_eb + theta_shear_lev

# Normalisation
w_max = max(np.max(np.abs(w_eb)), np.max(np.abs(w_tim)), np.max(np.abs(w_lev)))
theta_max = max(np.max(np.abs(theta_eb)), np.max(np.abs(theta_tim)), np.max(np.abs(theta_lev)))

w_eb_norm = w_eb / w_max
w_tim_norm = w_tim / w_max
w_lev_norm = w_lev / w_max

theta_eb_norm = theta_eb / theta_max
theta_tim_norm = theta_tim / theta_max
theta_lev_norm = theta_lev / theta_max

x_norm = x / L

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex='col')

# Unnormalised deflection (top-left)
axs[0, 0].plot(x, w_eb * 1e3, label="Euler-Bernoulli", linewidth=2)
axs[0, 0].plot(x, w_tim * 1e3, label="Timoshenko", linewidth=2)
axs[0, 0].plot(x, w_lev * 1e3, label="Levinson", linewidth=2)
axs[0, 0].set_ylabel("Deflection $w(x)$ [mm]")
axs[0, 0].set_title("Unnormalised Deflection")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Normalised deflection (top-right)
axs[0, 1].plot(x_norm, w_eb_norm, label="Euler-Bernoulli", linewidth=2)
axs[0, 1].plot(x_norm, w_tim_norm, label="Timoshenko", linewidth=2)
axs[0, 1].plot(x_norm, w_lev_norm, label="Levinson", linewidth=2)
axs[0, 1].set_title("Normalised Deflection")
axs[0, 1].grid(True)

# Unnormalised rotation (bottom-left)
axs[1, 0].plot(x, theta_eb * 1e3, label="Euler-Bernoulli", linewidth=2)
axs[1, 0].plot(x, theta_tim * 1e3, label="Timoshenko", linewidth=2)
axs[1, 0].plot(x, theta_lev * 1e3, label="Levinson", linewidth=2)
axs[1, 0].set_ylabel("Rotation $\\theta_z(x)$ [mrad]")
axs[1, 0].set_xlabel("Beam axis $x$ [m]")
axs[1, 0].set_title("Unnormalised Rotation")
axs[1, 0].grid(True)

# Normalised rotation (bottom-right)
axs[1, 1].plot(x_norm, theta_eb_norm, label="Euler-Bernoulli", linewidth=2)
axs[1, 1].plot(x_norm, theta_tim_norm, label="Timoshenko", linewidth=2)
axs[1, 1].plot(x_norm, theta_lev_norm, label="Levinson", linewidth=2)
axs[1, 1].set_xlabel("Normalised beam axis $x/L$")
axs[1, 1].set_title("Normalised Rotation")
axs[1, 1].grid(True)

# Final layout
plt.tight_layout()
plt.show()
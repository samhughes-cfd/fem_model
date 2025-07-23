import numpy as np
import matplotlib.pyplot as plt

# Beam and material properties (S275JR CoTide C-channel)
E = 2.1e11       # Young's modulus [Pa]
G = 8.1e10       # Shear modulus [Pa]
P = -500.0       # Point load [N] downward
L = 2.0          # Beam length [m]
b = 0.0131 / 0.1 # Assume initial area A=0.00131 with h=0.1 → b ≈ 0.131 m (constant width)

depths = [0.05, 0.1, 0.2]  # Varying depths [m]
kappa = 5/6

# Plot setup
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
colors = ['tab:blue', 'tab:green', 'tab:red']

for i, h in enumerate(depths):
    A = b * h                     # Cross-sectional area [m^2]
    I = (b * h**3) / 12           # Second moment of area [m^4]

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

    color = colors[i]
    linestyle = ['-', '--', ':']

    # Plot deflection
    axs[0].plot(x, w_eb * 1e3, linestyle[0], color=color, label=f"EB (h={h*1e3:.0f} mm)", linewidth=2)
    axs[0].plot(x, w_tim * 1e3, linestyle[1], color=color, label=f"Timo (h={h*1e3:.0f} mm)", linewidth=2)
    axs[0].plot(x, w_lev * 1e3, linestyle[2], color=color, label=f"Levinson (h={h*1e3:.0f} mm)", linewidth=2)

    # Plot rotation
    deg = 180 / np.pi
    axs[1].plot(x, theta_eb * deg, linestyle[0], color=color, linewidth=2)
    axs[1].plot(x, theta_tim * deg, linestyle[1], color=color, linewidth=2)
    axs[1].plot(x, theta_lev * deg, linestyle[2], color=color, linewidth=2)

# Labels and formatting
axs[0].set_ylabel("Deflection $w(x)$ [mm]")
axs[0].set_title("Effect of Depth on Deflection")
axs[0].legend(ncol=2, fontsize=9)
axs[0].grid(True)

axs[1].set_ylabel("Rotation $\\theta_z(x)$ [deg]")
axs[1].set_xlabel("Beam axis $x$ [m]")
axs[1].set_title("Effect of Depth on Rotation")
axs[1].grid(True)

plt.tight_layout()
plt.show()
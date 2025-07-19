import numpy as np
import matplotlib.pyplot as plt

# Analytical deflection at tip of a cantilever beam
def uy_tip(F, E, Iz, L):
    return F * L**3 / (3 * E * Iz)

# Base values
F_base = 500         # N
E_base = 2e11        # Pa
Iz_base = 2.08769e-6 # m^4
L_base = 1.0         # m

# Parameter ranges (log-spaced)
scales = np.logspace(-1, 1, 10)

# Containers for results
uy_F, uy_E, uy_Iz, uy_L = [], [], [], []

# --- Vary F ---
for scale in scales:
    uy_F.append(uy_tip(F_base * scale, E_base, Iz_base, L_base))
# --- Vary E ---
for scale in scales:
    uy_E.append(uy_tip(F_base, E_base * scale, Iz_base, L_base))
# --- Vary Iz ---
for scale in scales:
    uy_Iz.append(uy_tip(F_base, E_base, Iz_base * scale, L_base))
# --- Vary L ---
for scale in scales:
    uy_L.append(uy_tip(F_base, E_base, Iz_base, L_base * scale))

# Convert to arrays
uy_F = np.array(uy_F)
uy_E = np.array(uy_E)
uy_Iz = np.array(uy_Iz)
uy_L = np.array(uy_L)

# Compute log-log slopes (order of proportionality)
def loglog_slope(y_vals):
    log_x = np.log10(scales)
    log_y = np.log10(y_vals)
    slope, _ = np.polyfit(log_x, log_y, 1)
    return slope

slopes = {
    'F': loglog_slope(uy_F),
    'E': loglog_slope(uy_E),
    'Iz': loglog_slope(uy_Iz),
    'L': loglog_slope(uy_L),
}

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Top: Parametric Study (Linear Scale)
ax1.plot(scales, uy_F * 1000, 'o-', label='Vary F')
ax1.plot(scales, uy_E * 1000, 's-', label='Vary E')
ax1.plot(scales, uy_Iz * 1000, 'd-', label='Vary Iz')
ax1.plot(scales, uy_L * 1000, '^-', label='Vary L')
ax1.set_xscale('log')
ax1.set_ylabel(r"Tip Deflection $u_y(L)$ [mm]")
ax1.set_title("Parametric Study of Beam Deflection")
ax1.legend()
ax1.grid(True, which='both', linestyle='--', alpha=0.6)

# Bottom: Order of Proportionality (Log-Log)
ax2.plot(scales, uy_F, 'o-', label=f'F (order ≈ {slopes["F"]:.2f})')
ax2.plot(scales, uy_E, 's-', label=f'E (order ≈ {slopes["E"]:.2f})')
ax2.plot(scales, uy_Iz, 'd-', label=f'Iz (order ≈ {slopes["Iz"]:.2f})')
ax2.plot(scales, uy_L, '^-', label=f'L (order ≈ {slopes["L"]:.2f})')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel("Parameter scale (log10)")
ax2.set_ylabel(r"Tip Deflection $u_y(L)$ [m]")
ax2.set_title("Order of Proportionality via Log-Log Regression")
ax2.legend()
ax2.grid(True, which='both', linestyle='--', alpha=0.6)

fig.tight_layout()
plt.show()
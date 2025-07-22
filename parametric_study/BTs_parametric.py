import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from matplotlib.ticker import LogLocator

# --- Ensure plots directory exists ---
os.makedirs("parametric_study/plots", exist_ok=True)

# --- Beam theory functions ---
def uy_tip_eb(F, E, I, L):
    """Euler-Bernoulli tip deflection"""
    return F * L**3 / (3 * E * I)

def theta_tip_eb(F, E, I, L):
    """Euler-Bernoulli tip rotation"""
    return F * L**2 / (2 * E * I)

def uy_tip_timoshenko(F, E, I, L, G, A, kappa):
    """Timoshenko tip deflection (includes shear deformation)"""
    return uy_tip_eb(F, E, I, L) + F * L / (kappa * A * G)

def theta_tip_timoshenko(F, E, I, L):
    """Timoshenko tip rotation (same as Euler-Bernoulli)"""
    return theta_tip_eb(F, E, I, L)

def uy_tip_levinson(F, E, I, L, G, A):
    """Levinson tip deflection (includes shear deformation)"""
    return uy_tip_eb(F, E, I, L) + F * L / (G * A)

def theta_tip_levinson(F, E, I, L):
    """Levinson tip rotation (same as Euler-Bernoulli)"""
    return theta_tip_eb(F, E, I, L)

# --- Base parameters ---
F_base = 500         # Load [N]
E_base = 2.1e11      # Young's modulus [Pa]
G_base = 8.1e10      # Shear modulus [Pa]
I_base = 2.08769e-6  # Second moment of area [m⁴]
A_base = 0.00131     # Cross-sectional area [m²]
L_base = 1.0         # Beam length [m]
kappa = 5 / 6        # Shear correction factor

# --- Parameter space ---
scales = np.logspace(-1, 1, 10)  # Logarithmic scaling from 0.1 to 10
params = ['F', 'E', 'I', 'L']    # Parameters to vary
theories = ['eb', 'tim', 'lev']   # Beam theories (Euler-Bernoulli, Timoshenko, Levinson)

# --- Data containers ---
deflection_data = {t: {p: [] for p in params} for t in theories}
rotation_data = {t: {p: [] for p in params} for t in theories}
pi_deflection = {t: {p: [] for p in params} for t in theories}
pi_rotation = {t: {p: [] for p in params} for t in theories}

# --- Populate data arrays ---
for scale in scales:
    for theory in theories:
        for param in params:
            # Scale current parameter while keeping others at base values
            F = F_base * scale if param == 'F' else F_base
            E = E_base * scale if param == 'E' else E_base
            I = I_base * scale if param == 'I' else I_base
            L = L_base * scale if param == 'L' else L_base

            # Calculate deflection and rotation based on theory
            if theory == 'eb':
                uy = uy_tip_eb(F, E, I, L)
                theta = theta_tip_eb(F, E, I, L)
            elif theory == 'tim':
                uy = uy_tip_timoshenko(F, E, I, L, G_base, A_base, kappa)
                theta = theta_tip_timoshenko(F, E, I, L)
            elif theory == 'lev':
                uy = uy_tip_levinson(F, E, I, L, G_base, A_base)
                theta = theta_tip_levinson(F, E, I, L)

            deflection_data[theory][param].append(uy)
            rotation_data[theory][param].append(theta)

            # Calculate non-dimensional Pi groups
            pi_y = (F * L**3) / (E * I * uy)
            pi_theta = (F * L**2 * theta) / (E * I)
            
            pi_deflection[theory][param].append(pi_y)
            pi_rotation[theory][param].append(pi_theta)

# Convert to numpy arrays for analysis
for theory in theories:
    for param in params:
        deflection_data[theory][param] = np.array(deflection_data[theory][param])
        rotation_data[theory][param] = np.array(rotation_data[theory][param])
        pi_deflection[theory][param] = np.array(pi_deflection[theory][param])
        pi_rotation[theory][param] = np.array(pi_rotation[theory][param])

# --- Slope calculation for log-log plots ---
def loglog_slope(x, y):
    """Calculate slope of log-log data using linear regression"""
    log_x = np.log10(x)
    log_y = np.log10(y)
    slope, _ = np.polyfit(log_x, log_y, 1)
    return slope

# Compute slopes for all data
slopes_def = {t: {p: loglog_slope(scales, deflection_data[t][p]) for p in params} for t in theories}
slopes_rot = {t: {p: loglog_slope(scales, rotation_data[t][p]) for p in params} for t in theories}
slopes_pi_def = {t: {p: loglog_slope(scales, pi_deflection[t][p]) for p in params} for t in theories}
slopes_pi_rot = {t: {p: loglog_slope(scales, pi_rotation[t][p]) for p in params} for t in theories}

# --- Generate CSV reports ---
for theory in theories:
    csv_path = f"parametric_study/plots/deformation_summary_{theory}.csv"
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Theory", "Parameter", "Scale",
            "uy [m]", "theta_z [rad]",
            "Pi_uy", "Pi_theta",
            "Order_uy", "Order_theta",
            "Order_Pi_uy", "Order_Pi_theta"
        ])
        
        for param in params:
            for i, scale in enumerate(scales):
                writer.writerow([
                    theory, param, scale,
                    deflection_data[theory][param][i],
                    rotation_data[theory][param][i],
                    pi_deflection[theory][param][i],
                    pi_rotation[theory][param][i],
                    slopes_def[theory][param],
                    slopes_rot[theory][param],
                    slopes_pi_def[theory][param],
                    slopes_pi_rot[theory][param]
                ])
    print(f"✅ CSV written: {csv_path}")

# --- Generate log-log plots ---
for theory in theories:
    for param in params:
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f"Theory: {theory.upper()}, Parameter: {param}", fontsize=14)
        
        # Plot configuration
        titles = [
            "Tip Deflection (uy)",
            "Tip Rotation (θ)",
            "Non-dimensional Deflection (Π_uy)",
            "Non-dimensional Rotation (Π_θ)"
        ]
        ylabels = ["uy [m]", "θ [rad]", "Π_uy", "Π_θ"]
        data_sets = [
            deflection_data[theory][param],
            rotation_data[theory][param],
            pi_deflection[theory][param],
            pi_rotation[theory][param]
        ]
        slopes = [
            slopes_def[theory][param],
            slopes_rot[theory][param],
            slopes_pi_def[theory][param],
            slopes_pi_rot[theory][param]
        ]
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
        
        for i, ax in enumerate(axs):
            # Plot data points and line
            ax.loglog(scales, data_sets[i], 'o-', color=colors[i], 
                     markersize=6, linewidth=2, label='Data')
            
            # Add slope information
            slope_text = f'Slope: {slopes[i]:.3f}'
            ax.text(0.05, 0.95, slope_text, transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
            
            # Formatting
            ax.set_title(titles[i], fontsize=12)
            ax.set_ylabel(ylabels[i], fontsize=10)
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            ax.legend(loc='lower right')
            
            # Set custom x-ticks for better readability
            ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        
        axs[-1].set_xlabel("Scale Factor", fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"parametric_study/plots/{theory}_{param}_deformation.png", dpi=300)
        plt.close()

# --- Non-dimensional Characterization ---
print("\n=== Key Non-dimensional Parameters ===")

# 1. Slenderness Ratio
h_web = 0.1  # Web height [m]
slenderness = L_base / h_web
print(f"\n1. Slenderness ratio (L/h): {slenderness:.1f}")

# 2. Shear Influence Factor
shear_influence = (A_base * G_base * L_base**2) / (E_base * I_base)
print(f"2. Shear influence factor: {shear_influence:.3f}")

# 3. Flexural-to-Shear Rigidity Ratio
rigidity_ratio = (E_base * I_base) / (G_base * A_base * L_base**2)
print(f"3. Flexural-to-shear rigidity ratio: {rigidity_ratio:.5f}")
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Ensure plots directory exists ---
os.makedirs("parametric_study/plots", exist_ok=True)

# --- Beam theory functions ---
def uy_tip_eb(F, E, I, L):
    return F * L**3 / (3 * E * I)

def theta_tip_eb(F, E, I, L):
    return F * L**2 / (2 * E * I)

def uy_tip_timoshenko(F, E, I, L, G, A, kappa):
    return uy_tip_eb(F, E, I, L) + F * L / (kappa * A * G)

def theta_tip_timoshenko(F, E, I, L):
    return theta_tip_eb(F, E, I, L)

def uy_tip_levinson(F, E, I, L, G, A):
    return uy_tip_eb(F, E, I, L) + F * L / (G * A)

def theta_tip_levinson(F, E, I, L):
    return theta_tip_eb(F, E, I, L)

# --- Base parameters ---
F_base = 500               # N
E_base = 2.1e11            # Pa
G_base = 8.1e10            # Pa
I_base = 2.08769e-6        # m^4
A_base = 0.00131           # m^2
L_base = 1.0               # m
kappa = 5 / 6

# --- C-section dimensions (for context, not used in calc yet) ---
# Web depth (external):      100 mm
# Web thickness:             5 mm
# Flange width (external):   50 mm
# Flange thickness:          8.5 mm
# Internal corner radius:    10 mm

# --- Parameter scales ---
scales = np.logspace(-1, 1, 10)
params = ['F', 'E', 'I', 'L']
theories = ['eb', 'tim', 'lev']

# --- Containers ---
deflection_data = {t: {p: [] for p in params} for t in theories}
rotation_data = {t: {p: [] for p in params} for t in theories}
pi_deflection = {t: {p: [] for p in params} for t in theories}
pi_rotation = {t: {p: [] for p in params} for t in theories}

# --- Populate data ---
for scale in scales:
    for t in theories:
        for p in params:
            F = F_base * scale if p == 'F' else F_base
            E = E_base * scale if p == 'E' else E_base
            I = I_base * scale if p == 'I' else I_base
            L = L_base * scale if p == 'L' else L_base

            if t == 'eb':
                uy = uy_tip_eb(F, E, I, L)
                th = theta_tip_eb(F, E, I, L)
            elif t == 'tim':
                uy = uy_tip_timoshenko(F, E, I, L, G_base, A_base, kappa)
                th = theta_tip_timoshenko(F, E, I, L)
            elif t == 'lev':
                uy = uy_tip_levinson(F, E, I, L, G_base, A_base)
                th = theta_tip_levinson(F, E, I, L)

            deflection_data[t][p].append(uy)
            rotation_data[t][p].append(th)

            # Nondimensional groups
            pi_y = (F * L**3) / (E * I * uy)
            pi_th = (F * L**2 * th) / (E * I)
            pi_deflection[t][p].append(pi_y)
            pi_rotation[t][p].append(pi_th)

# Convert to arrays
for t in theories:
    for p in params:
        deflection_data[t][p] = np.array(deflection_data[t][p])
        rotation_data[t][p] = np.array(rotation_data[t][p])
        pi_deflection[t][p] = np.array(pi_deflection[t][p])
        pi_rotation[t][p] = np.array(pi_rotation[t][p])

# --- Log-log slope ---
def loglog_slope(y_vals):
    log_x = np.log10(scales)
    log_y = np.log10(y_vals)
    slope, _ = np.polyfit(log_x, log_y, 1)
    return slope

slopes_def = {t: {p: loglog_slope(deflection_data[t][p]) for p in params} for t in theories}
slopes_rot = {t: {p: loglog_slope(rotation_data[t][p]) for p in params} for t in theories}

# --- Plotting helper ---
def plot_response(data, slopes, ylabel, title, filename):
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex='col', sharey='row')
    theory_names = {'eb': 'Euler-Bernoulli', 'tim': 'Timoshenko', 'lev': 'Levinson'}
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for i, t in enumerate(theories):
        for j, p in enumerate(params):
            axs[i, 0].plot(scales, data[t][p], marker='o', color=colors[j],
                           label=f"{p} (order ≈ {slopes[t][p]:.2f})", linewidth=2)
        axs[i, 0].set_xscale('log')
        axs[i, 0].set_ylabel(f"{theory_names[t]}\n{ylabel}")
        axs[i, 0].grid(True, linestyle='--', alpha=0.5)
        if i == 0: axs[i, 0].set_title("Raw")

        for j, p in enumerate(params):
            axs[i, 1].plot(scales, data[t][p], marker='o', color=colors[j],
                           label=f"{p} (order ≈ {slopes[t][p]:.2f})", linewidth=2)
        axs[i, 1].set_xscale('log')
        axs[i, 1].set_yscale('log')
        axs[i, 1].grid(True, linestyle='--', alpha=0.5)
        if i == 0: axs[i, 1].set_title("Log-Log Scaling")

    axs[-1, 0].set_xlabel("Parameter scale (log10)")
    axs[-1, 1].set_xlabel("Parameter scale (log10)")
    axs[0, 1].legend(loc='upper right', fontsize=9)
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    plt.savefig(f"parametric_study/plots/{filename}", dpi=300)
    plt.close()

# --- Generate plots ---
plot_response(deflection_data, slopes_def, r"$u_y(L)$ [m]", "Tip Deflection Comparison", "tip_deflection.png")
plot_response(rotation_data, slopes_rot, r"$\theta_z(L)$ [rad]", "Tip Rotation Comparison", "tip_rotation.png")
plot_response(pi_deflection, slopes_def, r"$\Pi = \dfrac{F L^3}{EI u_y}$", "ND Tip Deflection", "pi_deflection.png")
plot_response(pi_rotation, slopes_rot, r"$\Pi_\theta = \dfrac{F L^2 \theta_z}{EI}$", "ND Tip Rotation", "pi_rotation.png")

# --- Slenderness and shear influence ---
h_web = 0.1       # m
L = L_base
slenderness = L / h_web
shear_influence = (A_base * G_base * L**2) / (E_base * I_base)
rigidity_ratio = 1 / shear_influence

print("\n=== Nondimensional Beam Characterisation ===")
print(f"Slenderness ratio (L/h):")
print(f"    Formula: L / h = {L:.3f} / {h_web:.3f} = {slenderness:.2f}")

print(f"\nShear influence factor:")
print("    Formula: (A·G·L²) / (E·I)")
print(f"    = ({A_base:.4e}·{G_base:.4e}·{L**2:.4e}) / ({E_base:.4e}·{I_base:.4e})")
print(f"    = {shear_influence:.2f}")

print(f"\nFlexural to shear rigidity ratio:")
print("    Formula: (E·I) / (A·G·L²)")
print(f"    = 1 / {shear_influence:.2f} = {rigidity_ratio:.5f}")

print("\nPlots saved to 'parametric_study/plots'")

# --- Future nondimensional parameters (not yet implemented) ---
# - Torsional stiffness group:             Π_T = T·L / G·J
# - Warping stiffness group:               Π_W = T·L / E·Cw
# - Dynamic scaling (1st freq):            Π_ω = ω·L² / sqrt(E·I / ρ·A)
# - Cross-section shape parameters:        warping constant (Cw), torsion constant (J)
# - Composite beam influence factors
# - Beam-column coupling (axial+flexural)
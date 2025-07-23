import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# --- Ensure plots directory exists ---
os.makedirs("parametric_study/plots", exist_ok=True)

# --- Beam theory functions ---
def uy_tip_eb(F, E, I, L):
    return F * L**3 / (3 * E * I)

def theta_tip(F, E, I, L):
    return F * L**2 / (2 * E * I)

def uy_tip_timoshenko(F, E, I, L, G, A, kappa):
    return uy_tip_eb(F, E, I, L) + F * L / (kappa * A * G)

def theta_tip_timoshenko(F, E, I, L):
    return theta_tip(F, E, I, L)

def uy_tip_levinson(F, E, I, L, G, A):
    return uy_tip_eb(F, E, I, L) + F * L / (G * A)

def theta_tip_levinson(F, E, I, L):
    return theta_tip(F, E, I, L)

# --- Base parameters (float64 enforced) ---
F_base = np.float64(500)
E_base = np.float64(2.1e11)
G_base = np.float64(8.1e10)
I_base = np.float64(2.08769e-6)
A_base = np.float64(0.00131)
L_base = np.float64(1.0)
kappa = np.float64(5 / 6)

# --- Parameter space ---
scales = np.logspace(-1, 1, 200, dtype=np.float64)
params = ['F', 'E', 'I', 'L']
theories = ['Euler-Bernoulli', 'Timoshenko', 'Levinson']

# --- Data containers ---
deflection_data = {t: {p: [] for p in params} for t in theories}
rotation_data = {t: {p: [] for p in params} for t in theories}
pi_deflection = {t: {p: [] for p in params} for t in theories}
pi_rotation = {t: {p: [] for p in params} for t in theories}

# --- Populate data arrays ---
for scale in scales:
    for theory in theories:
        for param in params:
            F = np.float64(F_base * scale if param == 'F' else F_base)
            E = np.float64(E_base * scale if param == 'E' else E_base)
            I = np.float64(I_base * scale if param == 'I' else I_base)
            L = np.float64(L_base * scale if param == 'L' else L_base)

            if theory == 'Euler-Bernoulli':
                uy = uy_tip_eb(F, E, I, L)
                theta = theta_tip(F, E, I, L)
            elif theory == 'Timoshenko':
                uy = uy_tip_timoshenko(F, E, I, L, G_base, A_base, kappa)
                theta = theta_tip_timoshenko(F, E, I, L)
            elif theory == 'Levinson':
                uy = uy_tip_levinson(F, E, I, L, G_base, A_base)
                theta = theta_tip_levinson(F, E, I, L)

            deflection_data[theory][param].append(uy)
            rotation_data[theory][param].append(theta)

            pi_y = (F * L**3) / (E * I * uy)
            pi_theta = (F * L**2 * theta) / (E * I)

            pi_deflection[theory][param].append(pi_y)
            pi_rotation[theory][param].append(pi_theta)

# --- Normalize Pi_uy and Pi_theta by baseline value ---
normalized_pi_deflection = {t: {p: None for p in params} for t in theories}
normalized_pi_rotation = {t: {p: None for p in params} for t in theories}
baseline_index = int(np.argmin(np.abs(scales - 1.0)))

for theory in theories:
    for param in params:
        pi_deflection[theory][param] = np.array(pi_deflection[theory][param], dtype=np.float64)
        pi_rotation[theory][param] = np.array(pi_rotation[theory][param], dtype=np.float64)
        deflection_data[theory][param] = np.array(deflection_data[theory][param], dtype=np.float64) * 1000.0  # mm
        rotation_data[theory][param] = np.degrees(np.array(rotation_data[theory][param], dtype=np.float64))   # deg

        baseline_uy = pi_deflection[theory][param][baseline_index]
        baseline_theta = pi_rotation[theory][param][baseline_index]

        normalized_pi_deflection[theory][param] = pi_deflection[theory][param] / baseline_uy
        normalized_pi_rotation[theory][param] = pi_rotation[theory][param] / baseline_theta

# --- Slope calculation ---
def loglog_slope(x, y):
    log_x = np.log10(x)
    log_y = np.log10(y)
    slope, _ = np.polyfit(log_x, log_y, 1)
    return slope

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
            "uy [mm]", "theta_z [deg]",
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

# --- Generate plots ---
for theory in theories:
    fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Theory: {theory.upper()} — Parametric Study", fontsize=14)

    # Titles for each subplot in (row, col) position
    titles = [
        ("Tip Deflection (uy)", "Tip Rotation (θ)"),
        ("Non-dimensional Deflection (Π_uy)", "Non-dimensional Rotation (Π_θ)"),
        ("Normalized Π_uy (Π_uy / Π_uy|scale=1)", "Normalized Π_θ (Π_θ / Π_θ|scale=1)")
    ]

    ylabels = [
        ("uy [mm]", "θ [deg]"),
        ("Π_uy", "Π_θ"),
        ("Π_uy norm", "Π_θ norm")
    ]

    all_data = [
        (deflection_data[theory], rotation_data[theory]),
        (pi_deflection[theory], pi_rotation[theory]),
        (normalized_pi_deflection[theory], normalized_pi_rotation[theory])
    ]

    all_slopes = [
        (slopes_def[theory], slopes_rot[theory]),
        (slopes_pi_def[theory], slopes_pi_rot[theory]),
        ({p: 0.0 for p in params}, {p: 0.0 for p in params})
    ]

    param_colors = {
        'F': 'tab:blue',
        'E': 'tab:orange',
        'I': 'tab:green',
        'L': 'tab:red'
    }

    for row in range(3):
        for col in range(2):
            ax = axs[row, col]
            data_dict = all_data[row][col]
            slope_dict = all_slopes[row][col]
            title = titles[row][col]
            ylabel = ylabels[row][col]

            for param in params:
                ax.plot(
                    scales,
                    data_dict[param],
                    '-',
                    color=param_colors[param],
                    linewidth=1,
                    label=f'{param}, slope={slope_dict[param]:.2f}'
                )

            ax.set_title(title, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            ax.legend(loc='best', fontsize=7)
            ax.set_xscale("log")
            ax.set_yscale("log")
            # Set major ticks at powers of 10
            ax.xaxis.set_major_locator(LogLocator(base=10.0))
            ax.xaxis.set_major_formatter(LogFormatterMathtext())  # Formats as 10^n

            # Optional: fine-tune minor ticks (optional)
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
            #ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Set shared x-label on the bottom row
    for ax in axs[2, :]:
        ax.set_xlabel("Scale Factor", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"parametric_study/plots/{theory}_combined_deformation.png", dpi=300)
    plt.close()

# --- Non-dimensional Characterization ---
print("\n=== Key Non-dimensional Parameters ===")

h_web = np.float64(0.1)
slenderness = L_base / h_web
print(f"\n1. Slenderness ratio (L/h): {slenderness:.1f}")

shear_influence = (A_base * G_base * L_base**2) / (E_base * I_base)
print(f"2. Shear influence factor: {shear_influence:.3f}")

rigidity_ratio = (E_base * I_base) / (G_base * A_base * L_base**2)
print(f"3. Flexural-to-shear rigidity ratio: {rigidity_ratio:.5f}")
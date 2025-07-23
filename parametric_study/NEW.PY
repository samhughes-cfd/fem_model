import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# --- Ensure plots directory exists ---
os.makedirs("parametric_study/plots", exist_ok=True)

# --- Beam theory functions ---
def uy_tip_eb(F, E, I, L):
    return F * L**3 / (3 * E * I)

def uy_bending(F, E, I, L):
    return F * L**3 / (3 * E * I)

def uy_shear_timoshenko(F, G, A, L, kappa):
    return F * L / (kappa * A * G)

def uy_shear_levinson(F, G, A, L):
    return F * L / (G * A)

# --- Base parameters ---
F = 500               # Load [N]
E = 2.1e11            # Young's modulus [Pa]
G = 8.1e10            # Shear modulus [Pa]
I = 2.08769e-6        # Second moment of area [m⁴]
A = 0.00131           # Cross-sectional area [m²]
kappa = 5 / 6         # Shear correction factor
h = 0.1               # Web height [m]

# --- Parameter space ---
L_vals = np.logspace(np.log10(0.05), np.log10(5.0), 50)  # Beam lengths [m]
slenderness_vals = L_vals / h

# --- Data containers ---
data_rows = []
uy_eb_list = []
uy_timo_list = []
uy_levinson_list = []

for L in L_vals:
    slenderness = L / h

    # Euler-Bernoulli
    uy_eb = uy_tip_eb(F, E, I, L)

    # Timoshenko
    uy_bend = uy_bending(F, E, I, L)
    uy_shear_timo = uy_shear_timoshenko(F, G, A, L, kappa)
    uy_timo = uy_bend + uy_shear_timo

    # Levinson
    uy_shear_lev = uy_shear_levinson(F, G, A, L)
    uy_lev = uy_bend + uy_shear_lev

    uy_eb_list.append(uy_eb)
    uy_timo_list.append(uy_timo)
    uy_levinson_list.append(uy_lev)

# --- Normalization ---
uy_eb_arr = np.array(uy_eb_list)
uy_timo_arr = np.array(uy_timo_list)
uy_levinson_arr = np.array(uy_levinson_list)

uy_bend_arr = uy_bending(F, E, I, L_vals)
uy_shear_timo_arr = uy_shear_timoshenko(F, G, A, L_vals, kappa)
uy_shear_lev_arr = uy_shear_levinson(F, G, A, L_vals)

uy_total_all = np.vstack([uy_eb_arr, uy_timo_arr, uy_levinson_arr])
uy_max = np.max(uy_total_all)

# --- Normalized arrays ---
uy_eb_norm = uy_eb_arr / uy_max
uy_timo_norm = uy_timo_arr / uy_max
uy_levinson_norm = uy_levinson_arr / uy_max
uy_bend_norm = uy_bend_arr / uy_max
uy_shear_timo_norm = uy_shear_timo_arr / uy_max
uy_shear_lev_norm = uy_shear_lev_arr / uy_max

# --- Save CSV ---
csv_path = "parametric_study/plots/decomposed_uy_vs_slenderness.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "L [m]", "L/h",
        "uy_EB", "uy_Timo", "uy_Levinson",
        "uy_bending", "uy_shear_Timo", "uy_shear_Levinson",
        "uy_EB_norm", "uy_Timo_norm", "uy_Levinson_norm",
        "uy_bending_norm", "uy_shear_Timo_norm", "uy_shear_Levinson_norm"
    ])
    for i in range(len(L_vals)):
        writer.writerow([
            L_vals[i], slenderness_vals[i],
            uy_eb_arr[i], uy_timo_arr[i], uy_levinson_arr[i],
            uy_bend_arr[i], uy_shear_timo_arr[i], uy_shear_lev_arr[i],
            uy_eb_norm[i], uy_timo_norm[i], uy_levinson_norm[i],
            uy_bend_norm[i], uy_shear_timo_norm[i], uy_shear_lev_norm[i]
        ])

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(slenderness_vals, uy_eb_norm, label="Euler-Bernoulli (Total)", linewidth=2)
plt.plot(slenderness_vals, uy_timo_norm, label="Timoshenko (Total)", linewidth=2)
plt.plot(slenderness_vals, uy_levinson_norm, label="Levinson (Total)", linewidth=2)
plt.plot(slenderness_vals, uy_bend_norm, '--', label="Bending Component", color='gray')
plt.plot(slenderness_vals, uy_shear_timo_norm, ':', label="Timoshenko Shear", color='tab:orange')
plt.plot(slenderness_vals, uy_shear_lev_norm, ':', label="Levinson Shear", color='tab:green')

plt.xlabel("Slenderness Ratio (L/h)")
plt.ylabel("Normalized Tip Deflection (uy / max(uy))")
plt.title("Decomposed Tip Deflection vs. Slenderness Ratio")
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.xscale('log')
plt.tight_layout()
plt.savefig("parametric_study/plots/decomposed_uy_vs_slenderness.png", dpi=300)
plt.close()

"✅ Script executed: plot and CSV for decomposed tip deflection generated."
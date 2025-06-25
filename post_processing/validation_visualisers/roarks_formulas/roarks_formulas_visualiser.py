# post_processing\validation_visualisers\deflection_tables\roarks_formulas_visualiser.py

import numpy as np
import matplotlib.pyplot as plt
import os
from roarks_formulas_point import RoarksFormulaePointLoad
from roarks_formulas_distributed import RoarksFormulaeDistributedLoad

class RoarkVisualiser:
    """
    Visualisation tool for Roark's beam solutions with consistent plotting conventions.
    
    Parameters:
        L (float): Beam length
        E (float): Young's modulus
        I (float): Area moment of inertia
        save_dir (str): Directory to save plots (default="roarks_formulas")
    """
    def __init__(self, L: float, E: float, I: float, save_dir: str = "roarks_formulas"):
        self.L = L
        self.E = E
        self.I = I
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Plotting configuration
        self.colors = {
            "intensity":  "#7F7F7F",  # gray
            "deflection": "#4F81BD",  # blue
            "rotation":   "#4F81BD",  # also blue
            "shear":      "#9BBB59",  # green
            "moment":     "#C0504D",  # red
        }
        
        self.plot_info = {
            "intensity": {
                "unit_factor": 1.0/1000.0,        # N -> kN or N/m -> kN/m
                "unit_name":   r"$kN \text{ or } kN/m$",
                "label_name":  r"$q(x)$",
            },
            "deflection": {
                "unit_factor": 1000.0,            # m -> mm
                "unit_name":   r"$mm$",
                "label_name":  r"$u_{y}(x)$",
            },
            "rotation": {
                "unit_factor": 180.0/np.pi,       # rad -> degrees
                "unit_name":   r"${}^{\circ}$",
                "label_name":  r"$\theta_{z}(x)$",
            },
            "shear": {
                "unit_factor": 1.0/1000.0,        # N -> kN
                "unit_name":   r"$kN$",
                "label_name":  r"$V(x)$",
            },
            "moment": {
                "unit_factor": 1.0/1000.0,        # N·m -> kN·m
                "unit_name":   r"$kN \cdot m$",
                "label_name":  r"$M(x)$",
            },
        }
    
    def convert_data(self, category, data):
        """Apply unit conversion to data"""
        factor = self.plot_info[category]["unit_factor"]
        return factor * data
    
    def plot_load_intensities(self, P, w, num_points=750, filename='q_intensities_2x3.png'):
        """
        Creates a 2x3 subplot figure for load intensities and saves as PNG
        """
        x_vals = np.linspace(0, self.L, num_points)
        point_load_types = ["end", "mid", "quarter"]
        dist_load_types = ["udl", "triangular", "parabolic"]
        
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), sharex=True)
        blue_color = "blue"
        
        # Point Loads (Row 0)
        for col_idx, lt in enumerate(point_load_types):
            ax = axes[0, col_idx]
            solver = RoarksFormulaePointLoad(self.L, self.E, self.I, P, load_type=lt)
            resp = solver.response(x_vals)
            
            q_vals = self.convert_data("intensity", resp["intensity"])
            idx_a = np.argmax(np.abs(q_vals))
            spike_x = x_vals[idx_a]
            spike_q = q_vals[idx_a]
            
            # Plot vertical force line
            ax.plot([spike_x, spike_x], [0, spike_q], color=blue_color, linewidth=2)
            ax.plot(spike_x, spike_q, marker="v", color=blue_color, markersize=8)
            ax.set_title(f"Point Load @ {lt.capitalize()}", fontsize=12)
            
            if col_idx == 0:
                ax.set_ylabel(r"$P(x)\,[kN]$")
            ax.grid(False)
        
        # Distributed Loads (Row 1)
        for col_idx, lt in enumerate(dist_load_types):
            ax = axes[1, col_idx]
            solver = RoarksFormulaeDistributedLoad(self.L, self.E, self.I, w, lt)
            resp = solver.response(x_vals)
            
            q_vals = self.convert_data("intensity", resp["intensity"])
            ax.plot(x_vals, q_vals, color=blue_color, linewidth=2)
            ax.fill_between(x_vals, q_vals, 0, color=blue_color, alpha=0.25)
            ax.set_title(f"{lt.capitalize()} Load", fontsize=12)
            
            if col_idx == 0:
                ax.set_ylabel(r"$q(x)\,[kN/m]$")
            ax.set_xlabel(r"$x\,[\mathrm{m}]$")
            ax.grid(False)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)  # Close the figure to free memory
        print(f"Saved load intensities plot: {save_path}")
    
    def plot_point_load_responses(self, P, num_points=200, filename='point_u_theta_V_M.png'):
        """
        Plots 4-row layout for point loads and saves as PNG
        """
        x_vals = np.linspace(0, self.L, num_points)
        load_types = ["end", "mid", "quarter"]
        
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12), sharex=True)
        
        for col_idx, lt in enumerate(load_types):
            solver = RoarksFormulaePointLoad(self.L, self.E, self.I, P, load_type=lt)
            resp = solver.response(x_vals)
            
            # Extract and convert responses
            u_vals = self.convert_data("deflection", resp["deflection"])
            th_vals = self.convert_data("rotation", resp["rotation"])
            V_vals = self.convert_data("shear", resp["shear"])
            M_vals = self.convert_data("moment", resp["moment"])
            
            # Deflection (row 0)
            ax = axes[0, col_idx]
            ax.plot(x_vals, u_vals, color=self.colors["deflection"])
            ax.set_title(f"Point Load @ {lt.capitalize()}", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(r"$u_{y}(x)\,[mm]$")
            
            # Rotation (row 1)
            ax = axes[1, col_idx]
            ax.plot(x_vals, th_vals, color=self.colors["rotation"])
            if col_idx == 0:
                ax.set_ylabel(r"$\theta_{z}(x)\,[^\circ]$")
            
            # Shear (row 2)
            ax = axes[2, col_idx]
            ax.step(x_vals, V_vals, where="post", color=self.colors["shear"])
            ax.fill_between(x_vals, V_vals, 0, step="post", 
                            color=self.colors["shear"], alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel(r"$V(x)\,[kN]$")
            
            # Moment (row 3)
            ax = axes[3, col_idx]
            ax.plot(x_vals, M_vals, color=self.colors["moment"])
            ax.fill_between(x_vals, M_vals, 0, color=self.colors["moment"], alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel(r"$M(x)\,[kN \cdot m]$")
            ax.set_xlabel(r"$x\,[\mathrm{m}]$")
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)  # Close the figure to free memory
        print(f"Saved point load responses plot: {save_path}")
    
    def plot_distributed_load_responses(self, w, num_points=750, filename='distributed_u_theta_V_M.png'):
        """
        Plots 4-row layout for distributed loads and saves as PNG
        """
        x_vals = np.linspace(0, self.L, num_points)
        load_types = ["udl", "triangular", "parabolic"]
        
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12), sharex=True)
        
        for col_idx, lt in enumerate(load_types):
            solver = RoarksFormulaeDistributedLoad(self.L, self.E, self.I, w, lt)
            resp = solver.response(x_vals)
            
            # Extract and convert responses
            u_vals = self.convert_data("deflection", resp["deflection"])
            th_vals = self.convert_data("rotation", resp["rotation"])
            V_vals = self.convert_data("shear", resp["shear"])
            M_vals = self.convert_data("moment", resp["moment"])
            
            # Deflection (row 0)
            ax = axes[0, col_idx]
            ax.plot(x_vals, u_vals, color=self.colors["deflection"])
            ax.set_title(f"{lt.capitalize()} Load", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(r"$u_{y}(x)\,[mm]$")
            
            # Rotation (row 1)
            ax = axes[1, col_idx]
            ax.plot(x_vals, th_vals, color=self.colors["rotation"])
            if col_idx == 0:
                ax.set_ylabel(r"$\theta_{z}(x)\,[^\circ]$")
            
            # Shear (row 2)
            ax = axes[2, col_idx]
            ax.plot(x_vals, V_vals, color=self.colors["shear"])
            ax.fill_between(x_vals, V_vals, 0, color=self.colors["shear"], alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel(r"$V(x)\,[kN]$")
            
            # Moment (row 3)
            ax = axes[3, col_idx]
            ax.plot(x_vals, M_vals, color=self.colors["moment"])
            ax.fill_between(x_vals, M_vals, 0, color=self.colors["moment"], alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel(r"$M(x)\,[kN \cdot m]$")
            ax.set_xlabel(r"$x\,[\mathrm{m}]$")
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)  # Close the figure to free memory
        print(f"Saved distributed load responses plot: {save_path}")

# ------------------------------------------------------------------------------
#   MAIN EXECUTION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Beam parameters
    L = 2.0            # [m]
    E = 2.0e11         # [Pa] (steel)
    I = 1.6667e-5      # [m⁴] (rectangular section)
    save_dir = "post_processing/validation_visualisers/roarks_formulas/plots"
    
    # Create visualizer
    visualizer = RoarkVisualiser(L, E, I, save_dir)
    
    # Load parameters
    P = 100000.0       # Point load [N]
    w = 100000.0       # Distributed load [N/m]
    
    # Generate plots and save as PNG
    visualizer.plot_load_intensities(P, w, filename='q_intensities_2x3.png')
    visualizer.plot_point_load_responses(P, filename='point_u_theta_V_M.png')
    visualizer.plot_distributed_load_responses(w, filename='distributed_u_theta_V_M.png')
    
    print("All plots saved successfully.")
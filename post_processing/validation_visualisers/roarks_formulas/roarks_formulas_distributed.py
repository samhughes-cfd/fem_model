# post_processing\validation_visualisers\deflection_tables\roarks_formulas._distriuted.py

import numpy as np
from scipy.integrate import cumulative_trapezoid

class RoarksFormulaeDistributedLoad:
    """
    Computes beam responses for distributed loads using Roark's formulas.
    
    Parameters:
        L (float): Beam length (must be >0)
        E (float): Young's modulus (must be >0)
        I (float): Area moment of inertia (must be >0)
        w (float): Load magnitude (positive downward)
        load_type (str): Load type - 'udl', 'triangular', or 'parabolic'
    """
    
    def __init__(self, L: float, E: float, I: float, w: float, load_type: str):
        self.L = L
        self.E = E
        self.I = I
        self.w = w
        self.load_type = load_type.lower()
        
        self.validate_parameters()
        
    def validate_parameters(self):
        """Check physical parameters and load type"""
        if self.L <= 0:
            raise ValueError("Beam length L must be positive")
        if self.E <= 0:
            raise ValueError("Young's modulus E must be positive")
        if self.I <= 0:
            raise ValueError("Moment of inertia I must be positive")
        if self.load_type not in ("udl", "triangular", "parabolic"):
            raise ValueError("load_type must be 'udl', 'triangular', or 'parabolic'")

    def intensity(self, x: np.ndarray) -> np.ndarray:
        """Distributed load intensity q(x)"""
        if self.load_type == "udl":
            return self.w * np.ones_like(x)
        elif self.load_type == "triangular":
            return self.w * (x / self.L)
        else:  # parabolic
            return self.w * (x / self.L)**2

    def shear(self, x: np.ndarray) -> np.ndarray:
        """Shear force distribution V(x)"""
        if self.load_type == "udl":
            return -self.w * (self.L - x)
        elif self.load_type == "triangular":
            return -self.w * (self.L**2 - x**2) / (2 * self.L)
        else:  # parabolic
            return -self.w * (self.L**3 - x**3) / (3 * self.L**2)

    def moment(self, x: np.ndarray) -> np.ndarray:
        """Bending moment distribution M(x)"""
        if self.load_type == "udl":
            return -self.w * (self.L - x)**2 / 2
        elif self.load_type == "triangular":
            return -self.w * (self.L - x)**2 * (2*self.L + x) / (6 * self.L)
        else:  # parabolic
            return -self.w * (self.L - x)**3 * (3*self.L + x) / (12 * self.L**2)

    def rotation(self, x: np.ndarray) -> np.ndarray:
        """Slope/rotation distribution θ_z(x) via cumulative integration"""
        if not np.all(np.diff(x) >= 0):
            raise ValueError("Input array x must be sorted in ascending order")
            
        M = self.moment(x)
        M_over_EI = M / (self.E * self.I)
        return cumulative_trapezoid(M_over_EI, x, initial=0)

    def deflection(self, x: np.ndarray) -> np.ndarray:
        """Deflection distribution u_y(x) via cumulative integration"""
        theta = self.rotation(x)
        return cumulative_trapezoid(theta, x, initial=0)

    def response(self, x: np.ndarray) -> dict:
        """
        Returns dictionary of beam responses:
        - intensity: q(x)
        - shear: V(x)
        - moment: M(x)
        - rotation: θ(x)
        - deflection: u(x)
        """
        return {
            "intensity": self.intensity(x),
            "shear": self.shear(x),
            "moment": self.moment(x),
            "rotation": self.rotation(x),
            "deflection": self.deflection(x)
        }
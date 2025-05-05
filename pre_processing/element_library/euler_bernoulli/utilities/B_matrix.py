# pre_processing\element_library\euler_bernoulli\utilities\B_matrix.py

import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class StrainDisplacementOperator:
    """
    Operator for strain-displacement relations in 3D Euler-Bernoulli beams.
    Provides rigorous transformation between natural (ξ) and physical (x) coordinates.

    Mathematical Formulation
    -----------------------
    Strain-Displacement Relations:
    - Axial strain: ε_x = du/dx = (∂N/∂ξ)(∂ξ/∂x)u
    - Bending curvatures: κ = d²w/dx² = (∂²N/∂ξ²)(∂ξ/∂x)²w
    - Torsional strain: φ_x = dθ_x/dx = (∂N/∂ξ)(∂ξ/∂x)θ_x

    Coordinate Transformation:
    - Natural to physical: x(ξ) = (1-ξ)/2 * x₁ + (1+ξ)/2 * x₂
    - Jacobian: J = dx/dξ = L/2
    - Derivatives: ∂ξ/∂x = 2/L, ∂²ξ/∂x² = 4/L²

    Parameters
    ----------
    element_length : float
        Physical length of element (x ∈ [0,L], L > 0)

    Attributes
    ----------
    jacobian : float
        Determinant of Jacobian matrix (dx/dξ = L/2)
    dξ_dx : float
        First derivative of coordinate transform (∂ξ/∂x = 2/L)
    d2ξ_dx2 : float
        Second derivative of coordinate transform (∂²ξ/∂x² = 4/L²)
    """

    element_length: float

    def __post_init__(self):
        """Precompute and validate transformation factors."""
        if self.element_length <= 0:
            raise ValueError(f"Element length must be positive, got {self.element_length}")
        
        object.__setattr__(self, '_jacobian', self.element_length / 2)
        object.__setattr__(self, '_dξ_dx', 2 / self.element_length)
        object.__setattr__(self, '_d2ξ_dx2', 4 / (self.element_length**2))

    @property
    def jacobian(self) -> float:
        """float: Jacobian determinant for integration (dx/dξ = L/2)"""
        return self._jacobian

    @property
    def dξ_dx(self) -> float:
        """float: First derivative of coordinate transform (∂ξ/∂x = 2/L)"""
        return self._dξ_dx

    @property 
    def d2ξ_dx2(self) -> float:
        """float: Second derivative of coordinate transform (∂²ξ/∂x² = 4/L²)"""
        return self._d2ξ_dx2

    def natural_coordinate_form(self, 
                              dN_dξ: np.ndarray, 
                              d2N_dξ2: np.ndarray) -> np.ndarray:
        """
        Constructs strain-displacement matrix in natural coordinates.

        Parameters
        ----------
        dN_dξ : np.ndarray [n_gauss, 12, 6]
            First derivatives of shape functions w.r.t. ξ
        d2N_dξ2 : np.ndarray [n_gauss, 12, 6]
            Second derivatives of shape functions w.r.t. ξ

        Returns
        -------
        np.ndarray [n_gauss, 4, 12]
            Strain-displacement matrix B̃ where ε̃ = B̃u in natural coordinates

        Notes
        -----
        Used for stiffness matrix computation:
        Kᵉ = ∫ B̃ᵀ D B̃ |J| dξ
        External scaling by Jacobian required during integration
        """
        B = np.empty((dN_dξ.shape[0], 4, 12))
        
        # Axial strain (ε_x)
        B[:, 0, [0,6]] = dN_dξ[:, [0,6], 0]  # ∂N/∂ξ
        
        # Bending-Z curvature (κ_z)
        B[:, 1, [1,7]] = d2N_dξ2[:, [1,7], 1]  # ∂²N/∂ξ²
        B[:, 1, [5,11]] = d2N_dξ2[:, [5,11], 5]  # Rotation coupling
        
        # Bending-Y curvature (κ_y)
        B[:, 2, [2,8]] = d2N_dξ2[:, [2,8], 2]
        B[:, 2, [4,10]] = d2N_dξ2[:, [4,10], 4]
        
        # Torsional strain (φ_x)
        B[:, 3, [3,9]] = dN_dξ[:, [3,9], 3]
        
        return B

    def physical_coordinate_form(self,
                               dN_dξ: np.ndarray,
                               d2N_dξ2: np.ndarray) -> np.ndarray:
        """
        Constructs strain-displacement matrix in physical coordinates.

        Parameters
        ----------
        dN_dξ : np.ndarray [n_gauss, 12, 6]
            First derivatives of shape functions w.r.t. ξ
        d2N_dξ2 : np.ndarray [n_gauss, 12, 6]
            Second derivatives of shape functions w.r.t. ξ

        Returns
        -------
        np.ndarray [n_gauss, 4, 12]
            Strain-displacement matrix B where ε = Bu in physical coordinates

        Notes
        -----
        Used for strain recovery and post-processing:
        ε = B u
        All coordinate transforms are applied internally
        """
        B = np.empty((dN_dξ.shape[0], 4, 12))
        
        # Transform axial strain (ε_x = (∂N/∂ξ)(∂ξ/∂x))
        B[:, 0, [0,6]] = dN_dξ[:, [0,6], 0] * self.dξ_dx
        
        # Transform bending curvatures (κ = (∂²N/∂ξ²)(∂ξ/∂x)²)
        B[:, 1, [1,7]] = d2N_dξ2[:, [1,7], 1] * self.d2ξ_dx2
        B[:, 1, [5,11]] = d2N_dξ2[:, [5,11], 5] * self.dξ_dx
        
        B[:, 2, [2,8]] = d2N_dξ2[:, [2,8], 2] * self.d2ξ_dx2
        B[:, 2, [4,10]] = d2N_dξ2[:, [4,10], 4] * self.dξ_dx
        
        # Transform torsional strain (φ_x = (∂N/∂ξ)(∂ξ/∂x))
        B[:, 3, [3,9]] = dN_dξ[:, [3,9], 3] * self.dξ_dx
        
        return B

    def verify_coordinate_transforms(self, tol: float = 1e-12) -> Tuple[bool, str]:
        """
        Validates consistency of coordinate transformations.

        Parameters
        ----------
        tol : float
            Tolerance for numerical comparisons

        Returns
        -------
        Tuple[bool, str]
            (True, "All transformations valid") if checks pass,
            (False, error_message) otherwise
        """
        checks = {
            'Jacobian': abs(self.jacobian - self.element_length/2),
            'First derivative': abs(self.dξ_dx*self.element_length - 2),
            'Second derivative': abs(self.d2ξ_dx2*(self.element_length**2) - 4)
        }
        for name, error in checks.items():
            if error > tol:
                return False, f"{name} transform error: {error:.2e}"
        return True, "All transformations valid"
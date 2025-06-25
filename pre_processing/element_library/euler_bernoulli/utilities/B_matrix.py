# pre_processing\element_library\euler_bernoulli\utilities\B_matrix.py

import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class StrainDisplacementOperator:
    """
    Builds the strain–displacement matrix **B** for a 2-node 3-D Euler–Bernoulli beam element.

    The operator transforms derivatives of shape functions into physical strain measures.

    Strain vector:
        ε = [ ε_x  κ_y  κ_z  γ_xy  γ_xz  φ_x ]ᵀ

    where:
        ε_x   = ∂uₓ/∂x        (axial)
        κ_y   = ∂²w /∂x²      (bending about y, x–z plane)
        κ_z   = ∂²v /∂x²      (bending about z, x–y plane)
        γ_xy  = 0             (shear xy not modelled in Euler-Bernoulli theory)
        γ_xz  = 0             (shear xz not modelled in Euler-Bernoulli theory)
        φ_x   = ∂θₓ/∂x        (torsion)

    Coordinate mapping:
        - x(ξ) = ((1 - ξ)/2)x₁ + ((1 + ξ)/2)x₂
        - dx/dξ = L/2 ⇒ ∂ξ/∂x = 2/L
        - ∂²ξ/∂x² = 4/L²

    Parameters
    ----------
    element_length : float
        Length `L` of the beam element (must be > 0)

    Attributes
    ----------
    jacobian : float
        Jacobian of coordinate mapping (L/2)
    dξ_dx : float
        First derivative ∂ξ/∂x (2/L)
    d2ξ_dx2 : float
        Second derivative ∂²ξ/∂x² (4/L²)
    """

    element_length: float

    def __post_init__(self):
        if self.element_length <= 0:
            raise ValueError(f"Element length must be positive, got {self.element_length}")
        object.__setattr__(self, '_jacobian', self.element_length / 2)
        object.__setattr__(self, '_dξ_dx', 2 / self.element_length)
        object.__setattr__(self, '_d2ξ_dx2', 4 / self.element_length ** 2)

    @property
    def jacobian(self) -> float:
        """float: Jacobian of isoparametric mapping (dx/dξ = L/2)"""
        return self._jacobian

    @property
    def dξ_dx(self) -> float:
        """float: First derivative ∂ξ/∂x = 2/L"""
        return self._dξ_dx

    @property
    def d2ξ_dx2(self) -> float:
        """float: Second derivative ∂²ξ/∂x² = 4/L²"""
        return self._d2ξ_dx2

    def natural_coordinate_form(self,
                                dN_dξ: np.ndarray,
                                d2N_dξ2: np.ndarray) -> np.ndarray:
        """
        Construct strain-displacement matrix `B̃` in natural coordinates (ξ-space).

        Parameters
        ----------
        dN_dξ : np.ndarray (n_gauss, 12, 6)
            First derivatives of shape functions
        d2N_dξ2 : np.ndarray (n_gauss, 12, 6)
            Second derivatives of shape functions

        Returns
        -------
        B : np.ndarray (n_gauss, 6, 12)
            Strain-displacement matrix in ξ-space
        """
        n_gauss = dN_dξ.shape[0]
        B = np.zeros((n_gauss, 6, 12))

        # Axial strain: ε_x = ∂u_x/∂ξ
        B[:, 0, [0, 6]] = dN_dξ[:, [0, 6], 0]       # u_x

        # Bending about y-axis: κ_y = ∂²u_z/∂ξ²
        B[:, 1, [2, 8]] = d2N_dξ2[:, [2, 8], 2]     # u_z
        B[:, 1, [4, 10]] = d2N_dξ2[:, [4, 10], 4]   # θ_y

        # Bending about z-axis: κ_z = ∂²u_y/∂ξ²
        B[:, 2, [1, 7]] = d2N_dξ2[:, [1, 7], 1]     # u_y
        B[:, 2, [5, 11]] = d2N_dξ2[:, [5, 11], 5]   # θ_z

        # Torsional strain: φ_x = ∂θ_x/∂ξ
        B[:, 5, [3, 9]] = dN_dξ[:, [3, 9], 3]       # θ_x

        return B

    def physical_coordinate_form(self,
                                 dN_dξ: np.ndarray,
                                 d2N_dξ2: np.ndarray) -> np.ndarray:
        """
        Construct strain-displacement matrix `B` in physical coordinates (x-space).

        Parameters
        ----------
        dN_dξ : np.ndarray (n_gauss, 12, 6)
            First derivatives of shape functions
        d2N_dξ2 : np.ndarray (n_gauss, 12, 6)
            Second derivatives of shape functions

        Returns
        -------
        B : np.ndarray (n_gauss, 6, 12)
            Physical strain-displacement matrix (ε = B @ u_e)
        """
        n_gauss = dN_dξ.shape[0]
        B = np.zeros((n_gauss, 6, 12))

        # Axial strain: ε_x = ∂u_x/∂x
        B[:, 0, [0, 6]] = dN_dξ[:, [0, 6], 0] * self.dξ_dx           # u_x

        # Bending about y-axis: κ_y = ∂²u_z/∂ξ²
        B[:, 1, [2, 8]] = d2N_dξ2[:, [2, 8], 2] * self.d2ξ_dx2       # u_z
        B[:, 1, [4, 10]] = d2N_dξ2[:, [4, 10], 4] * self.d2ξ_dx2     # θ_y
 
        # Bending about z-axis: κ_z = ∂²u_y/∂ξ²
        B[:, 2, [1, 7]] = d2N_dξ2[:, [1, 7], 1] * self.d2ξ_dx2       # u_y
        B[:, 2, [5, 11]] = d2N_dξ2[:, [5, 11], 5] * self.d2ξ_dx2     # θ_z

        # Torsional strain: φ_x = ∂θ_x/∂ξ
        B[:, 5, [3, 9]] = dN_dξ[:, [3, 9], 3] * self.dξ_dx           # θ_x

        return B

    def verify_coordinate_transforms(self, tol: float = 1e-12) -> Tuple[bool, str]:
        """
        Validate coordinate transformation parameters.

        Parameters
        ----------
        tol : float
            Numerical tolerance for validation

        Returns
        -------
        Tuple[bool, str]
            Validation status and message
        """
        checks = [
            ("Jacobian", abs(self.jacobian - self.element_length/2)),
            ("∂ξ/∂x", abs(self.dξ_dx - 2/self.element_length)),
            ("∂²ξ/∂x²", abs(self.d2ξ_dx2 - 4/self.element_length**2))
        ]
        for name, error in checks:
            if error > tol:
                return False, f"{name} error: {error:.2e} > {tol}"
        return True, "All transforms valid"
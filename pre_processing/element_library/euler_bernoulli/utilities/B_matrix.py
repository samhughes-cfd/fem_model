# pre_processing\element_library\euler_bernoulli\utilities\B_matrix.py

import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class StrainDisplacementOperator:
    """
    Constructs the strain-displacement matrix `B` for a 2-node 3D Euler-Bernoulli beam element.

    The operator transforms derivatives of shape functions into physical strain measures.

    Strain vector:
        ε = [εₓ, κ_z, κ_y, φₓ]ᵀ

    where:
        - εₓ  = ∂uₓ/∂x          (axial strain)
        - κ_z = ∂²v/∂x²         (curvature about z-axis, x-y plane bending)
        - κ_y = ∂²w/∂x²         (curvature about y-axis, x-z plane bending)
        - φₓ  = ∂θₓ/∂x          (torsional strain)

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
        B : np.ndarray (n_gauss, 4, 12)
            Strain-displacement matrix in ξ-space
        """
        B = np.zeros((dN_dξ.shape[0], 4, 12))

        # Axial strain: εₓ = ∂uₓ/∂ξ
        B[:, 0, [0, 6]] = dN_dξ[:, [0, 6], 0]

        # Bending about z-axis: κ_z = ∂²v/∂ξ²
        B[:, 1, [1, 7]] = d2N_dξ2[:, [1, 7], 1]

        # Bending about y-axis: κ_y = ∂²w/∂ξ²
        B[:, 2, [2, 8]] = d2N_dξ2[:, [2, 8], 2]

        # Torsional strain: φₓ = ∂θₓ/∂ξ
        B[:, 3, [3, 9]] = dN_dξ[:, [3, 9], 3]

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
        B : np.ndarray (n_gauss, 4, 12)
            Physical strain-displacement matrix (ε = B @ u_e)
        """
        B = np.zeros((dN_dξ.shape[0], 4, 12))

        # Axial strain: εₓ = ∂uₓ/∂x
        B[:, 0, [0, 6]] = dN_dξ[:, [0, 6], 0] * self.dξ_dx

        # Bending about z-axis: κ_z = ∂²v/∂x²
        B[:, 1, [1, 7]] = d2N_dξ2[:, [1, 7], 1] * self.d2ξ_dx2

        # Bending about y-axis: κ_y = ∂²w/∂x²
        B[:, 2, [2, 8]] = d2N_dξ2[:, [2, 8], 2] * self.d2ξ_dx2

        # Torsional strain: φₓ = ∂θₓ/∂x
        B[:, 3, [3, 9]] = dN_dξ[:, [3, 9], 3] * self.dξ_dx

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
            ("First derivative", abs(self.dξ_dx - 2/self.element_length)),
            ("Second derivative", abs(self.d2ξ_dx2 - 4/self.element_length**2))
        ]
        for name, error in checks:
            if error > tol:
                return False, f"{name} error: {error:.2e} > {tol}"
        return True, "All transforms valid"
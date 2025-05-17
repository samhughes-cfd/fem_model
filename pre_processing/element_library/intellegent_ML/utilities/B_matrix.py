# pre_processing\element_library\euler_bernoulli\utilities\B_matrix.py

import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class StrainDisplacementOperator:
    """
    Constructs the strain-displacement matrix `B` for a 3D Euler-Bernoulli beam element.

    The operator transforms first and second derivatives of shape functions with respect 
    to the natural coordinate ξ ∈ [-1, 1] into physical strain measures in x ∈ [0, L]. 

    Strain vector:
        ε = [εₓ, κ_z, κ_y, φₓ]ᵀ

    where:
        - εₓ  = ∂uₓ/∂x          (axial strain)
        - κ_z = ∂²w/∂x² + ∂²θ_y/∂x²   (curvature due to bending in x–y plane)
        - κ_y = ∂²v/∂x² + ∂²θ_z/∂x²   (curvature due to bending in x–z plane)
        - φₓ  = ∂θₓ/∂x          (torsional strain)

    Coordinate mapping:
        - x(ξ) = ((1 - ξ) / 2) * x₁ + ((1 + ξ) / 2) * x₂
        - dx/dξ = L/2 ⇒ ∂ξ/∂x = 2/L
        - ∂²ξ/∂x² = 4 / L²

    Parameters
    ----------
    element_length : float
        Length `L` of the beam element in the global x-direction (must be > 0).

    Attributes
    ----------
    jacobian : float
        Determinant of the isoparametric mapping: dx/dξ = L / 2

    dξ_dx : float
        First derivative of ξ with respect to x: ∂ξ/∂x = 2 / L

    d2ξ_dx2 : float
        Second derivative of ξ with respect to x: ∂²ξ/∂x² = 4 / L²
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
        """float: Jacobian of isoparametric mapping (dx/dξ = L / 2)"""
        return self._jacobian

    @property
    def dξ_dx(self) -> float:
        """float: First derivative ∂ξ/∂x = 2 / L"""
        return self._dξ_dx

    @property
    def d2ξ_dx2(self) -> float:
        """float: Second derivative ∂²ξ/∂x² = 4 / L²"""
        return self._d2ξ_dx2

    def natural_coordinate_form(self,
                                dN_dξ: np.ndarray,
                                d2N_dξ2: np.ndarray) -> np.ndarray:
        """
        Construct strain-displacement matrix `B̃` in natural coordinates (ξ-space).

        Parameters
        ----------
        dN_dξ : ndarray of shape (n_gauss, 12, 6)
            First derivatives ∂N/∂ξ of shape functions with respect to ξ.
        d2N_dξ2 : ndarray of shape (n_gauss, 12, 6)
            Second derivatives ∂²N/∂ξ² of shape functions with respect to ξ.

        Returns
        -------
        B : ndarray of shape (n_gauss, 4, 12)
            Strain-displacement matrix in ξ-space, used before transformation to physical space.

        Notes
        -----
        This form is used for symbolic verification and internal consistency checks.
        Curvatures include contributions from both displacement and rotational DOFs.
        """
        B = np.zeros((dN_dξ.shape[0], 4, 12))

        # Axial strain εₓ = ∂uₓ/∂ξ
        B[:, 0, [0, 6]] = dN_dξ[:, [0, 6], 0]

        # Bending curvature κ_z = ∂²w/∂ξ² + ∂²θ_y/∂ξ²
        B[:, 1, [2, 8]] = d2N_dξ2[:, [2, 8], 2]
        B[:, 1, [4, 10]] = d2N_dξ2[:, [4, 10], 4]

        # Bending curvature κ_y = ∂²v/∂ξ² + ∂²θ_z/∂ξ²
        B[:, 2, [1, 7]] = d2N_dξ2[:, [1, 7], 1]
        B[:, 2, [5, 11]] = d2N_dξ2[:, [5, 11], 5]

        # Torsional strain φₓ = ∂θₓ/∂ξ
        B[:, 3, [3, 9]] = dN_dξ[:, [3, 9], 3]

        return B

    def physical_coordinate_form(self,
                                 dN_dξ: np.ndarray,
                                 d2N_dξ2: np.ndarray) -> np.ndarray:
        """
        Construct strain-displacement matrix `B` in physical coordinates (x-space).

        Parameters
        ----------
        dN_dξ : ndarray of shape (n_gauss, 12, 6)
            First derivatives ∂N/∂ξ of shape functions with respect to ξ.
        d2N_dξ2 : ndarray of shape (n_gauss, 12, 6)
            Second derivatives ∂²N/∂ξ² of shape functions with respect to ξ.

        Returns
        -------
        B : ndarray of shape (n_gauss, 4, 12)
            Physical strain-displacement matrix such that ε = B @ u_e

        Notes
        -----
        - The coordinate transformation is handled internally.
        - Curvatures (κ_z, κ_y) include second derivatives of both translation and rotation DOFs.
        """
        B = np.zeros((dN_dξ.shape[0], 4, 12))

        # εₓ = ∂uₓ/∂x = ∂uₓ/∂ξ * ∂ξ/∂x
        B[:, 0, [0, 6]] = dN_dξ[:, [0, 6], 0] * self.dξ_dx

        # κ_z = ∂²w/∂x² + ∂²θ_y/∂x²
        B[:, 1, [2, 8]] = d2N_dξ2[:, [2, 8], 2] * self.d2ξ_dx2
        B[:, 1, [4, 10]] = d2N_dξ2[:, [4, 10], 4] * self.d2ξ_dx2

        # κ_y = ∂²v/∂x² + ∂²θ_z/∂x²
        B[:, 2, [1, 7]] = d2N_dξ2[:, [1, 7], 1] * self.d2ξ_dx2
        B[:, 2, [5, 11]] = d2N_dξ2[:, [5, 11], 5] * self.d2ξ_dx2

        # φₓ = ∂θₓ/∂x
        B[:, 3, [3, 9]] = dN_dξ[:, [3, 9], 3] * self.dξ_dx

        return B

    def verify_coordinate_transforms(self, tol: float = 1e-12) -> Tuple[bool, str]:
        """
        Check analytical coordinate transform identities within tolerance.

        Parameters
        ----------
        tol : float, optional
            Numerical tolerance for validation. Default is 1e-12.

        Returns
        -------
        Tuple[bool, str]
            (True, message) if valid; otherwise (False, error message).
        """
        checks = [
            ("Jacobian", abs(self.jacobian - self.element_length / 2)),
            ("First derivative", abs(self.dξ_dx - 2 / self.element_length)),
            ("Second derivative", abs(self.d2ξ_dx2 - 4 / self.element_length ** 2))
        ]
        for name, error in checks:
            if error > tol:
                return False, f"{name} transform error: {error:.2e} > {tol}"
        return True, "All coordinate transforms valid"
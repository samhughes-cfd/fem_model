import numpy as np
from typing import Tuple, Literal, Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class ShapeFunctionOperator:
    """
    Operator for evaluating Euler-Bernoulli beam shape functions and their derivatives 
    in both natural (ξ) and physical (x) coordinates.

    Mathematical Definitions:
        Natural coordinates: ξ ∈ [-1, 1]
        Physical coordinates: x ∈ [0, L]
        Mapping: x(ξ) = (1 - ξ)/2 * x₁ + (1 + ξ)/2 * x₂
        Jacobian: J = dx/dξ = L/2

    Shape Function Types:
        1. Axial (Linear Lagrange):
            N₁(ξ) = 0.5(1 - ξ), N₇(ξ) = 0.5(1 + ξ)
        2. Bending (Hermite Cubic):
            Displacement: N₂(ξ) = 1 - 3ξ² + 2ξ³, N₈(ξ) = 3ξ² - 2ξ³
            Rotation: N₆(ξ) = ξ - 2ξ² + ξ³, N₁₂(ξ) = -ξ² + ξ³
        3. Torsion (Linear Lagrange):
            N₄(ξ) = 0.5(1 - ξ), N₁₀(ξ) = 0.5(1 + ξ)
    """

    L: float = 1.0  # Element length in physical coordinates
    _dξ_dx: float = None  # 2/L (derivative transform)
    _d2ξ_dx2: float = None  # 4/L² (second derivative transform)

    def __post_init__(self):
        """Precompute coordinate transformation factors."""
        object.__setattr__(self, '_dξ_dx', 2 / self.L)
        object.__setattr__(self, '_d2ξ_dx2', 4 / self.L**2)

    def __call__(self, 
                ξ: np.ndarray,
                mode: Literal['natural', 'physical'] = 'natural',
                return_derivatives: bool = True) -> Tuple[np.ndarray, ...]:
        """
        Evaluate shape functions and derivatives at given natural coordinates.

        Parameters
        ----------
        ξ : np.ndarray
            Natural coordinates ∈ [-1, 1] with shape (n_points,)
        mode : {'natural', 'physical'}
            Coordinate system for returned derivatives:
            - 'natural': Derivatives w.r.t. ξ (d/dξ, d²/dξ²)
            - 'physical': Derivatives w.r.t. x (d/dx, d²/dx²)
        return_derivatives : bool
            Whether to compute and return derivatives

        Returns
        -------
        N : np.ndarray
            Shape function matrix with shape (n_points, 12, 6)
        dN : np.ndarray, optional
            First derivative matrix (n_points, 12, 6)
        d2N : np.ndarray, optional
            Second derivative matrix (n_points, 12, 6)

        Notes
        -----
        Matrix organization follows standard FEM DOF ordering:
        Node 1: [u_x, u_y, u_z, θ_x, θ_y, θ_z]
        Node 2: [u_x, u_y, u_z, θ_x, θ_y, θ_z]
        """
        ξ = np.asarray(ξ, dtype=np.float64)
        n_points = ξ.size
        ξ = ξ.reshape(-1, 1, 1)  # Prepare for broadcasting

        # Compute in natural coordinates
        N, dN_dξ, d2N_dξ2 = self._evaluate_natural(ξ, n_points)
        
        if mode == 'physical':
            # Transform to physical coordinates
            dN_dξ *= self._dξ_dx
            d2N_dξ2 *= self._d2ξ_dx2

        return (N, dN_dξ, d2N_dξ2) if return_derivatives else (N,)

    def _evaluate_natural(self, 
                         ξ: np.ndarray, 
                         n_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Core polynomial evaluation in natural coordinates.

        Parameters
        ----------
        ξ : np.ndarray
            Reshaped natural coordinates with shape (n_points, 1, 1)
        n_points : int
            Number of evaluation points

        Returns
        -------
        N : np.ndarray
            Shape functions (n_points, 12, 6)
        dN_dξ : np.ndarray
            First derivatives w.r.t. ξ (n_points, 12, 6)
        d2N_dξ2 : np.ndarray
            Second derivatives w.r.t. ξ (n_points, 12, 6)
        """
        # Polynomial basis [1, ξ, ξ², ξ³]
        P = np.array([np.ones_like(ξ), ξ, ξ**2, ξ**3])  # shape (4, n_points, 1, 1)

        # Initialize output arrays
        N = np.zeros((n_points, 12, 6))
        dN_dξ = np.zeros_like(N)
        d2N_dξ2 = np.zeros_like(N)

        # ----- Axial Terms (Linear Lagrange) -----
        # N₁(ξ) = 0.5(1 - ξ), N₇(ξ) = 0.5(1 + ξ)
        N[:, [0,6], 0] = 0.5 * np.array([1 - ξ.squeeze(), 1 + ξ.squeeze()]).T
        dN_dξ[:, [0,6], 0] = 0.5 * np.array([-1, 1])

        # ----- Bending in XY Plane (Hermite Cubic) -----
        # Displacement terms
        N[:, [1,7], 1] = np.array([1 - 3*ξ**2 + 2*ξ**3, 
                                   3*ξ**2 - 2*ξ**3]).squeeze().T
        dN_dξ[:, [1,7], 1] = np.array([-6*ξ + 6*ξ**2, 
                                       6*ξ - 6*ξ**2]).squeeze().T
        d2N_dξ2[:, [1,7], 1] = np.array([-6 + 12*ξ, 
                                         6 - 12*ξ]).squeeze().T

        # Rotation terms
        N[:, [5,11], 5] = np.array([ξ - 2*ξ**2 + ξ**3, 
                                    -ξ**2 + ξ**3]).squeeze().T
        dN_dξ[:, [5,11], 5] = np.array([1 - 4*ξ + 3*ξ**2, 
                                       -2*ξ + 3*ξ**2]).squeeze().T
        d2N_dξ2[:, [5,11], 5] = np.array([-4 + 6*ξ, 
                                         -2 + 6*ξ]).squeeze().T

        # ----- Bending in XZ Plane (Hermite Cubic) -----
        # Displacement terms (mirror XY plane)
        N[:, [2,8], 2] = N[:, [1,7], 1]
        dN_dξ[:, [2,8], 2] = dN_dξ[:, [1,7], 1]
        d2N_dξ2[:, [2,8], 2] = d2N_dξ2[:, [1,7], 1]

        # Rotation terms (negative sign convention)
        N[:, [4,10], 4] = -N[:, [5,11], 5]
        dN_dξ[:, [4,10], 4] = -dN_dξ[:, [5,11], 5]
        d2N_dξ2[:, [4,10], 4] = -d2N_dξ2[:, [5,11], 5]

        # ----- Torsion Terms (Linear Lagrange) -----
        N[:, [3,9], 3] = N[:, [0,6], 0]
        dN_dξ[:, [3,9], 3] = dN_dξ[:, [0,6], 0]

        return N, dN_dξ, d2N_dξ2

    @property
    def dof_interpretation(self) -> np.ndarray:
        """
        Structured array documenting DOF physical meaning.

        Returns
        -------
        np.ndarray
            Structured array with fields:
            - index: DOF index (0-11)
            - node: 'Node 1' or 'Node 2'
            - component: 'u_x', 'u_y', 'u_z', 'θ_x', 'θ_y', 'θ_z'
            - behavior: 'Axial', 'Bending XY', 'Bending XZ', 'Torsion'
        """
        return np.array([
            (0, 'Node 1', 'u_x', 'Axial'),
            (1, 'Node 1', 'u_y', 'Bending XY'),
            (2, 'Node 1', 'u_z', 'Bending XZ'),
            (3, 'Node 1', 'θ_x', 'Torsion'),
            (4, 'Node 1', 'θ_y', 'Bending XZ'), 
            (5, 'Node 1', 'θ_z', 'Bending XY'),
            (6, 'Node 2', 'u_x', 'Axial'),
            (7, 'Node 2', 'u_y', 'Bending XY'),
            (8, 'Node 2', 'u_z', 'Bending XZ'),
            (9, 'Node 2', 'θ_x', 'Torsion'),
            (10, 'Node 2', 'θ_y', 'Bending XZ'),
            (11, 'Node 2', 'θ_z', 'Bending XY')
        ], dtype=[('index', 'i4'), ('node', 'U10'), ('component', 'U3'), ('behavior', 'U10')])
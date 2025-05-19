# pre_processing\element_library\timoshenko\utilities\shape_functions.py

import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class ShapeFunctionOperator:
    """
    Operator for 3D Timoshenko beam shape functions with independent rotational DOFs 
    and shear deformation.

    Mathematical Formulation
    -----------------------
    DOF Interpolation:
    - Axial displacement (u_x): 
      Linear Lagrange polynomials (O(ξ^1))
        N1_u = 0.5(1 - ξ), N2_u = 0.5(1 + ξ)
    
    - Transverse displacements (u_y, u_z):
      Cubic Hermitian polynomials (O(ξ^3)) with C1 continuity:
        N1_v = 0.25(2 - 3ξ + ξ^3)
        N2_v = 0.25(2 + 3ξ - ξ^3)
    
    - Rotations (θ_y, θ_z):
      Quadratic Lagrange polynomials (O(ξ^2)) with C0 continuity:
        N1_θ = 0.25(1 - ξ^2)
        N2_θ = 0.25(ξ^2 - 1)
    
    - Torsional rotation (θ_x):
      Linear Lagrange polynomials (O(ξ^1)) same as axial displacement

    Strain Formulation:
    - Axial strain: ε_xx = du_x/dx
    - Shear strains: γ_xy = du_y/dx - θ_z, γ_xz = du_z/dx - θ_y
    - Curvatures: κ_x = dθ_x/dx, κ_y = dθ_y/dx, κ_z = dθ_z/dx

    Element Characteristics:
    - 2 nodes, 6 DOFs per node (u_x, u_y, u_z, θ_x, θ_y, θ_z)
    - Mixed interpolation: Cubic (u_t) + Quadratic (θ) prevents shear locking
    - Consistent with Timoshenko beam theory (constant shear strain assumption)

    Coordinate Transformation:
    - Identical to Euler-Bernoulli: ξ = (2x - L)/L
    - Derivatives follow chain rule: ∂/∂x = (2/L)∂/∂ξ
    """

    element_length: float

    def __post_init__(self):
        if self.element_length <= 0:
            raise ValueError(f"Element length must be positive, got {self.element_length}")
        object.__setattr__(self, '_dξ_dx', 2 / self.element_length)
        object.__setattr__(self, '_d2ξ_dx2', 4 / (self.element_length**2))

    @property
    def dξ_dx(self) -> float:
        return self._dξ_dx

    @property
    def d2ξ_dx2(self) -> float:
        return self._d2ξ_dx2

    def natural_coordinate_form(self, ξ: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ξ = np.asarray(ξ, dtype=np.float64).reshape(-1, 1, 1)
        n_points = ξ.size
        N = np.zeros((n_points, 12, 6))
        dN_dξ = np.zeros_like(N)
        d2N_dξ2 = np.zeros_like(N)

        # ----- Axial Displacement (Linear) -----
        N[:, [0,6], 0] = 0.5 * np.array([1 - ξ.squeeze(), 1 + ξ.squeeze()]).T
        dN_dξ[:, [0,6], 0] = 0.5 * np.array([-1, 1])

        # ----- Bending XY Plane (Cubic u_y, Quadratic θ_z) -----
        # u_y: Cubic Hermite
        N[:, [1,7], 1] = np.array([1 - 3*ξ**2 + 2*ξ**3, 3*ξ**2 - 2*ξ**3]).squeeze().T
        dN_dξ[:, [1,7], 1] = np.array([-6*ξ + 6*ξ**2, 6*ξ - 6*ξ**2]).squeeze().T
        d2N_dξ2[:, [1,7], 1] = np.array([-6 + 12*ξ, 6 - 12*ξ]).squeeze().T
        
        # θ_z: Quadratic Lagrange
        N[:, [5,11], 5] = 0.25 * np.array([1 - ξ**2, ξ**2 - 1]).squeeze().T
        dN_dξ[:, [5,11], 5] = 0.5 * np.array([-ξ, ξ]).squeeze().T
        d2N_dξ2[:, [5,11], 5] = 0.5 * np.array([-1, 1])

        # ----- Bending XZ Plane (Cubic u_z, Quadratic θ_y) -----
        N[:, [2,8], 2] = N[:, [1,7], 1]  # Mirror u_y
        dN_dξ[:, [2,8], 2] = dN_dξ[:, [1,7], 1]
        d2N_dξ2[:, [2,8], 2] = d2N_dξ2[:, [1,7], 1]
        
        # θ_y: Quadratic Lagrange (sign convention matches rotation direction)
        N[:, [4,10], 4] = -N[:, [5,11], 5]
        dN_dξ[:, [4,10], 4] = -dN_dξ[:, [5,11], 5]
        d2N_dξ2[:, [4,10], 4] = -d2N_dξ2[:, [5,11], 5]

        # ----- Torsional Rotation (Linear) -----
        N[:, [3,9], 3] = N[:, [0,6], 0]
        dN_dξ[:, [3,9], 3] = dN_dξ[:, [0,6], 0]

        return N, dN_dξ, d2N_dξ2

    def physical_coordinate_form(self, ξ: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N, dN_dξ, d2N_dξ2 = self.natural_coordinate_form(ξ)
        dN_dx = dN_dξ * self.dξ_dx
        d2N_dx2 = d2N_dξ2 * self.d2ξ_dx2
        return N, dN_dx, d2N_dx2

    @property
    def dof_interpretation(self) -> np.ndarray:
        return np.array([
            (0, 'Node 1', 'u_x', 'Axial'),
            (1, 'Node 1', 'u_y', 'Bending XY'),
            (2, 'Node 1', 'u_z', 'Bending XZ'),
            (3, 'Node 1', 'θ_x', 'Torsion'),
            (4, 'Node 1', 'θ_y', 'Bending XZ (Independent)'), 
            (5, 'Node 1', 'θ_z', 'Bending XY (Independent)'),
            (6, 'Node 2', 'u_x', 'Axial'),
            (7, 'Node 2', 'u_y', 'Bending XY'),
            (8, 'Node 2', 'u_z', 'Bending XZ'),
            (9, 'Node 2', 'θ_x', 'Torsion'),
            (10, 'Node 2', 'θ_y', 'Bending XZ (Independent)'),
            (11, 'Node 2', 'θ_z', 'Bending XY (Independent)')
        ], dtype=[('index', 'i4'), ('node', 'U10'), ('component', 'U3'), ('behavior', 'U20')])
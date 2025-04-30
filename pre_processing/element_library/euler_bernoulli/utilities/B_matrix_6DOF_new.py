# pre_processing\element_library\euler_bernoulli\utilities\B_matrix_6DOF.py

import numpy as np
from typing import Literal, Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class StrainDisplacementOperator:
    """
    Computational mechanics operator for Euler-Bernoulli beam strain-displacement relations.
    Provides rigorous transformation between natural and physical coordinates.

    Mathematical Foundations:
    - Isoparametric mapping: x(ξ) = ∑ N_i(ξ)x_i
    - Strain measures:
        ε_x = du/dx = (∂N/∂ξ)(dξ/dx)u
        κ = d²u/dx² = (∂²N/∂ξ²)(dξ/dx)²u
    """
    
    element_length: float  # Physical length (x ∈ [0,L])
    jacobian: float = None  # dx/dξ
    dξ_dx: float = None  # First derivative transform
    d2ξ_dx2: float = None  # Second derivative transform
    
    def __post_init__(self):
        """Precompute transformation factors upon initialization"""
        object.__setattr__(self, 'jacobian', self.element_length / 2)
        object.__setattr__(self, 'dξ_dx', 2 / self.element_length)
        object.__setattr__(self, 'd2ξ_dx2', 4 / self.element_length**2)
    
    def compute_B_matrix(self, 
                       dN_dξ: np.ndarray, 
                       d2N_dξ2: np.ndarray,
                       mode: Literal['stiffness', 'strain'] = 'stiffness') -> np.ndarray:
        """
        Compute the strain-displacement matrix with rigorous coordinate transformation.

        Args:
            dN_dξ: First derivatives of shape functions [n_gauss, 12, 6]
            d2N_dξ2: Second derivatives of shape functions [n_gauss, 12, 6]
            mode: 
                'stiffness' - For stiffness matrix integration (K = ∫ B̃ᵀ D B̃ dξ)
                'strain' - For physical strain recovery (ε = B u)
                
        Returns:
            B_matrix: Strain-displacement matrices [n_gauss, 4, 12]
        """
        B = np.empty((dN_dξ.shape[0], 4, 12))
        
        if mode == 'stiffness':
            # Stiffness integration form (natural coordinates)
            B[:, 0, [0,6]] = dN_dξ[:, [0,6], 0]  # Axial
            B[:, 1, [1,7]] = d2N_dξ2[:, [1,7], 1] / self.jacobian  # Bending-Z (displacement)
            B[:, 1, [5,11]] = d2N_dξ2[:, [5,11], 5]  # Bending-Z (rotation)
            B[:, 2, [2,8]] = d2N_dξ2[:, [2,8], 2] / self.jacobian  # Bending-Y (displacement)
            B[:, 2, [4,10]] = d2N_dξ2[:, [4,10], 4]  # Bending-Y (rotation)
            B[:, 3, [3,9]] = dN_dξ[:, [3,9], 3]  # Torsion
        else:
            # Physical strain form (physical coordinates)
            B[:, 0, [0,6]] = dN_dξ[:, [0,6], 0] * self.dξ_dx
            B[:, 1, [1,7]] = d2N_dξ2[:, [1,7], 1] * self.d2ξ_dx2
            B[:, 1, [5,11]] = d2N_dξ2[:, [5,11], 5] * self.dξ_dx
            B[:, 2, [2,8]] = d2N_dξ2[:, [2,8], 2] * self.d2ξ_dx2
            B[:, 2, [4,10]] = d2N_dξ2[:, [4,10], 4] * self.dξ_dx
            B[:, 3, [3,9]] = dN_dξ[:, [3,9], 3] * self.dξ_dx
            
        return B

    @property
    def integration_jacobian(self) -> float:
        """Returns the Jacobian determinant for numerical integration: ∫f(x)dx = ∫f(ξ)|J|dξ"""
        return self.jacobian

    def verify_transformations(self, tol: float = 1e-12) -> Tuple[bool, str]:
        """Validate coordinate transformation consistency"""
        checks = {
            'Jacobian': abs(self.jacobian - self.element_length/2),
            'First derivative': abs(self.dξ_dx*self.element_length - 2),
            'Second derivative': abs(self.d2ξ_dx2*(self.element_length**2) - 4)
        }
        for name, error in checks.items():
            if error > tol:
                return False, f"{name} transform error: {error:.2e}"
        return True, "All transformations valid"

# Usage Example
#strain_op = StrainDisplacementOperator(element_length=2.0)

# Stiffness matrix computation
#Ke = sum(
    #B.T @ material.D @ B * weight * strain_op.integration_jacobian
    #for B in (strain_op.compute_B_matrix(dN_dξ, d2N_dξ2, 'stiffness')
    #for dN_dξ, d2N_dξ2, weight in gauss_quadrature_data
#)

# Strain recovery
#strain = strain_op.compute_B_matrix(dN_dξ_sample, d2N_dξ2_sample, 'strain') @ nodal_displacements
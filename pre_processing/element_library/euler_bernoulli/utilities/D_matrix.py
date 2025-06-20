# pre_processing\element_library\euler_bernoulli\utilities\D_matrix.py

import numpy as np
from typing import Dict
from dataclasses import dataclass, field

@dataclass(frozen=True)
class MaterialStiffnessOperator:
    """Constitutive operator for 3D Euler-Bernoulli beam elements.
    
    Encapsulates the material stiffness matrix (D-matrix) with dual representations:
    - Assembly form: Optimized for stiffness matrix assembly (Kᵉ = ∫BᵀDB dx)
    - Postprocessing form: Complete form for stress/strain computation and energy decomposition

    Mathematical Formulation
    -----------------------
    The constitutive relation follows Euler-Bernoulli beam theory:
    
    ⎡ N  ⎤   ⎡ EA     0       0        0   ⎤ ⎡ ε_x ⎤
    ⎢ M_z⎥ = ⎢ 0     EI_z     0        0   ⎥ ⎢ κ_z ⎥
    ⎢ M_y⎥   ⎢ 0      0     EI_y       0   ⎥ ⎢ κ_y ⎥
    ⎣ M_x⎦   ⎣ 0      0       0     GJ_t ⎦ ⎣ φ_x ⎦

    Parameters
    ----------
    youngs_modulus : float
        Young's modulus (E) in Pascals (Pa)
    shear_modulus : float
        Shear modulus (G) in Pascals (Pa)
    cross_section_area : float
        Cross-sectional area (A) in m²
    moment_inertia_y : float
        Second moment of area about y-axis (I_y) in m⁴
    moment_inertia_z : float
        Second moment of area about z-axis (I_z) in m⁴
    torsion_constant : float
        Torsional constant (J_t) in m⁴

    Attributes
    ----------
    """

    # Material properties (immutable)
    youngs_modulus: float
    shear_modulus: float
    cross_section_area: float
    moment_inertia_y: float
    moment_inertia_z: float
    torsion_constant: float

    # Internal matrices
    _D_assembly: np.ndarray = field(init=False, repr=False)
    _D_postprocess: np.ndarray = field(init=False, repr=False)
    _energy_components: Dict[str, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self):
        """Validate properties and build matrices immediately after construction."""
        self._validate_properties()
        self._build_constitutive_matrices()

    def assembly_form(self) -> np.ndarray:
        """
        Retrieves the material matrix optimized for stiffness matrix assembly.
        
        Used in the computation of Kᵉ = ∫BᵀDB dx where:
        - B is the strain-displacement matrix
        - D is this material stiffness matrix
        - Integration is performed over element domain

        Returns
        -------
        np.ndarray
            4×4 material stiffness matrix in assembly-optimized form
        """
        return self._D_assembly

    def postprocessing_form(self) -> np.ndarray:
        """
        Retrieves the complete material matrix for analysis and visualization.
        
        Used for:
        - Stress recovery (σ = Dε)
        - Strain energy calculations
        - Result verification and postprocessing

        Returns
        -------
        np.ndarray 
            4×4 material stiffness matrix in complete form
        """
        return self._D_postprocess

    def compute_stress_resultants(self, strain: np.ndarray) -> np.ndarray:
        """
        Compute stress resultants from strain measures using full constitutive relation.
        
        Parameters
        ----------
        strain : np.ndarray, shape (4,) or (4,n)
            Strain vector/matrix in Voigt notation [ε_x, κ_z, κ_y, φ_x]

        Returns
        -------
        np.ndarray
            Stress resultants [N, M_z, M_y, M_x] in same shape as input
        """
        return self.postprocessing_form() @ strain

    def energy_density_components(self, strain: np.ndarray) -> Dict[str, float]:
        """
        Decomposes strain energy density by deformation mode.
        
        Returns
        -------
        Dict[str, float]
            Components with keys:
            - 'total' : Total strain energy density
            - 'axial' : Axial deformation energy
            - 'bending_z' : Bending about z-axis energy
            - 'bending_y' : Bending about y-axis energy  
            - 'torsion' : Torsional energy
        """
        return {
            'total': 0.5 * strain.T @ self._D_postprocess @ strain,
            **{k: 0.5 * strain.T @ v @ strain 
               for k,v in self._energy_components.items()}
        }

    def _validate_properties(self) -> None:
        """Verify physical plausibility of all material parameters."""
        if not all(x > 0 for x in [
            self.youngs_modulus, self.shear_modulus,
            self.cross_section_area, self.moment_inertia_y,
            self.moment_inertia_z, self.torsion_constant
        ]):
            raise ValueError("All stiffness parameters must be strictly positive")

    def _build_constitutive_matrices(self) -> None:
        """Constructs and validates all constitutive matrices."""
        # Compute stiffness terms (consistent units)
        EA = self.youngs_modulus * self.cross_section_area
        EI_z = self.youngs_modulus * self.moment_inertia_z
        EI_y = self.youngs_modulus * self.moment_inertia_y
        GJ_t = self.shear_modulus * self.torsion_constant

        # Construct base matrix (diagonal for Euler-Bernoulli)
        D = np.array([
            [EA, 0, 0, 0],        # Axial
            [0, EI_z, 0, 0],      # Bending-Z
            [0, 0, EI_y, 0],      # Bending-Y
            [0, 0, 0, GJ_t]       # Torsion
        ], dtype=np.float64)

        object.__setattr__(self, '_D_assembly', D)
        object.__setattr__(self, '_D_postprocess', D.copy())

        object.__setattr__(self, '_energy_components', {
            'axial': np.diag([EA, 0, 0, 0]),
            'bending_z': np.diag([0, EI_z, 0, 0]),
            'bending_y': np.diag([0, 0, EI_y, 0]),
            'torsion': np.diag([0, 0, 0, GJ_t])
        })
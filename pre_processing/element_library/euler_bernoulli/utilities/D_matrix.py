# pre_processing\element_library\euler_bernoulli\utilities\D_matrix.py

import numpy as np
from typing import Dict
from dataclasses import dataclass, field

@dataclass(frozen=True)
class MaterialStiffnessOperator:
    """
    Constitutive operator for 3-D Euler–Bernoulli beam elements.

    This class builds and stores the element-level material stiffness
    tensor (**D-matrix**) in two convenient forms:

    * **assembly form** – a matrix tailored for the stiffness-integration
      loop : `Kᵉ = ∫ Bᵀ D B dx`
    * **post-processing form** – the same matrix kept for stress recovery,
      energy checks, result visualisation, etc.

    Mathematical Formulation
    ------------------------
    Classical Euler–Bernoulli theory couples axial, bending and torsional
    actions while shear terms remain identically zero:

        ε = [ εₓ  κᵧ  κ_z  γ_xy  γ_xz  φₓ ]ᵀ
        N = [ N   Mᵧ  M_z  V_xy  V_xz  Mₓ ]ᵀ

                       ⎡ EA    0     0     0     0     0 ⎤
                       ⎢  0   EI_y   0     0     0     0 ⎥
    N = D · ε  ,   D = ⎢  0    0   EI_z    0     0     0 ⎥
                       ⎢  0    0     0     0     0     0 ⎥
                       ⎢  0    0     0     0     0     0 ⎥
                       ⎣  0    0     0     0     0   GJ_t⎦

        EA   = E · A       (axial stiffness)  
        EI_y = E · I_y     (bending about y)  
        EI_z = E · I_z     (bending about z)  
        GJ_t = G · J_t     (torsional stiffness)

    Parameters
    ----------
    youngs_modulus : float
        Young’s modulus *E* [Pa].
    shear_modulus : float
        Shear modulus *G* [Pa].
    cross_section_area : float
        Cross-sectional area *A* [m²].
    moment_inertia_y : float
        Second moment of area about **y** (*I_y*) [m⁴].
    moment_inertia_z : float
        Second moment of area about **z** (*I_z*) [m⁴].
    torsion_constant : float
        Torsional constant *J_t* [m⁴].

    Attributes
    ----------
    _D_assembly : ndarray (6 × 6)
        Sparse-by-design matrix used inside the element stiffness loop.
    _D_postprocess : ndarray (6 × 6)
        Copy of *D* kept intact for stress / energy work.
    _energy_components : dict[str, ndarray]
        Pre-factored diagonal blocks for axial, bending-y, bending-z,
        torsion, shear-xy (zero) and shear-xz (zero).

    Public API
    ----------
    assembly_form()
        Return the assembly-optimised D-matrix, 6×6 matrix used in Kᵉ = ∫ Bᵀ D B |J| dξ
    postprocessing_form()
        Return the full post-processing D-matrix, identical copy, kept for symmetry with other
        element types that may differ  
    compute_stress_resultants(ε)
        σ = D · ε for one or many strain vectors.
    energy_density_components(ε)
        Detailed strain-energy breakdown.

    """

    # --- Material properties (immutable) ---
    youngs_modulus: float
    shear_modulus: float
    cross_section_area: float
    moment_inertia_y: float
    moment_inertia_z: float
    torsion_constant: float

    # --- Internal container matrices --- 

    _D_assembly: np.ndarray = field(init=False, repr=False)
    _D_postprocess: np.ndarray = field(init=False, repr=False)
    _energy_components: Dict[str, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self):
        """Validate properties and build internal matrices immediately after construction."""
        self._validate_properties()
        self._build_constitutive_matrices()

    def _validate_properties(self) -> None:
        if not all(x > 0 for x in (
            self.youngs_modulus,
            self.shear_modulus,
            self.cross_section_area,
            self.moment_inertia_y,
            self.moment_inertia_z,
            self.torsion_constant,
        )):
            raise ValueError("All stiffness parameters must be strictly positive")

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
            6×6 material stiffness matrix in assembly-optimized form
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
            6×6 material stiffness matrix in complete form
        """
        return self._D_postprocess

    def compute_stress_resultants(self, strain: np.ndarray) -> np.ndarray:
        """
        Compute stress resultants from strain measures using full constitutive relation.
        
        Parameters
        ----------
        strain : np.ndarray, shape (6,) or (6,n)
            Strain vector/matrix in Voigt notation ε = [ εₓ  κᵧ  κ_z  γ_xy  γ_xz  φₓ ]ᵀ, γ_xy = γ_xz= 0

        Returns
        -------
        np.ndarray`
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

    def _build_constitutive_matrices(self) -> None:
        """Constructs and validates all constitutive matrices.
        
        | Voigt component | Meaning                    | Row/col in D | Filled with |
        | --------------- | -------------------------- | -------------| ----------- |
        | 0               | ε_x  (axial)               | 0            | **EA**      |
        | 1               | κ_y (bending about y)      | 1            | **EI\_y**   |
        | 2               | κ_z (bending about z)      | 2            | **EI\_z**   |
        | 3               | γ_xy (shear-xy)            | 3            | 0           |
        | 4               | γ_xz (shear-xz)            | 4            | 0           |
        | 5               | φ_x  (torsion)             | 5            | **GJ\_t**   |

        """
        # Compute stiffness terms (consistent units)
        EA = self.youngs_modulus * self.cross_section_area
        EI_z = self.youngs_modulus * self.moment_inertia_z
        EI_y = self.youngs_modulus * self.moment_inertia_y
        GJ_t = self.shear_modulus * self.torsion_constant

        # --- main constitutive matrix ---
        D = np.zeros((6, 6), dtype=np.float64)
        D[0, 0] = EA
        D[1, 1] = EI_y          # bending about y
        D[2, 2] = EI_z          # bending about z
        D[5, 5] = GJ_t          # torsion

        object.__setattr__(self, '_D_assembly', D)
        object.__setattr__(self, '_D_postprocess', D.copy())

        object.__setattr__(self, '_energy_components', {
            'axial'     : np.diag([EA, 0, 0, 0, 0, 0]),
            'bending_y' : np.diag([0, EI_y, 0, 0, 0, 0]),
            'bending_z' : np.diag([0, 0, EI_z, 0, 0, 0]),
            'torsion'   : np.diag([0, 0, 0, 0, 0, GJ_t]),
            'shear_xy'  : np.zeros((6, 6)),
            'shear_xz'  : np.zeros((6, 6)),
        })
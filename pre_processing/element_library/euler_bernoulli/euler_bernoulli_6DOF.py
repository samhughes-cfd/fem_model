# pre_processing\element_library\euler_bernoulli\euler_bernoulli_6DOF.py

import numpy as np
from typing import Tuple

# Import Element1DBase class
from pre_processing.element_library.element_1D_base import Element1DBase

# Import MaterialStiffnessOperator, ShapeFunctionOperator and StrainDisplacementOperator classes
from pre_processing.element_library.euler_bernoulli.utilities.D_matrix import MaterialStiffnessOperator
from pre_processing.element_library.euler_bernoulli.utilities.shape_functions import ShapeFunctionOperator
from pre_processing.element_library.euler_bernoulli.utilities.B_matrix import StrainDisplacementOperator

# Import LoadInterpolationOperator class
from pre_processing.element_library.euler_bernoulli.utilities.interpolate_loads import LoadInterpolationOperator

class EulerBernoulliBeamElement6DOF(Element1DBase):
    """
    2-node 3D Euler-Bernoulli Beam Element with full matrix computation capabilities
    
    Features:
    - Exact shape function implementation
    - Configurable quadrature order
    - Combined point/distributed load handling
    - Property-based access to material/geometry parameters
    """
    
    def __init__(self, 
                 geometry_array: np.ndarray,
                 material_array: np.ndarray,
                 mesh_dictionary: dict,
                 point_load_array: np.ndarray,
                 distributed_load_array: np.ndarray,
                 element_id: int,
                 quadrature_order: int = 3):
        """
        Initialize a 6-DOF beam element

        Args:
            geometry_array: Geometry properties array [1x20]
            material_array: Material properties array [1x4]
            mesh_dictionary: Mesh data dictionary
            point_load_array: Point load array [Nx9]
            distributed_load_array: Distributed load array [Nx9]
            element_id: Element ID in the mesh
            quadrature_order: Integration order (default=3)
        """
        super().__init__(geometry_array, material_array, mesh_dictionary,
                         point_load_array, distributed_load_array, dof_per_node=6)
        
        self.element_id = element_id
        self.quadrature_order = quadrature_order
    
        # Initialize element geometry
        self._init_element_geometry()
        self._validate_element_properties()
    
        # Initialize the three operator classes
        self.shape_function_operator = ShapeFunctionOperator(element_length=self.L)
        self.strain_displacement_operator = StrainDisplacementOperator(element_length=self.L)
        self.material_stiffness_operator = MaterialStiffnessOperator(
            youngs_modulus=self.E,
            shear_modulus=self.G,
            cross_section_area=self.A,
            moment_inertia_y=self.I_y, 
            moment_inertia_z=self.I_z,
            torsion_constant=self.J_t,
            warping_inertia_y=self.warping_inertia_y,
            warping_inertia_z=self.warping_inertia_z
        )

    def _init_element_geometry(self) -> None:
        """Initialize element-specific geometric properties"""
        conn = self.mesh_dictionary["connectivity"][self.element_id]
        self.node_coords = self.mesh_dictionary["node_coordinates"][conn]
        
        self.x_start = self.node_coords[0, 0]
        self.x_end = self.node_coords[1, 0]
        self.L = self.x_end - self.x_start
        
        # Get global X maximum for boundary condition handling
        self.x_global_end = np.max(self.mesh_dictionary["node_coordinates"][:, 0])

    def _validate_element_properties(self) -> None:
        """Validate critical element properties"""
        if self.L <= 0:
            raise ValueError(f"Invalid element length {self.L:.2e} for element {self.element_id}")
        if not hasattr(self, 'material_array') or self.material_array.size == 0:
            raise ValueError("Material properties not properly initialized")
        if self.geometry_array.shape != (1, 20):
            raise ValueError("Geometry array must have shape (1, 20)")

    # Property definitions -----------------------------------------------------
    @property
    def A(self) -> float:
        """Cross-sectional area (m²)"""
        return self.geometry_array[0, 1]
    
    @property
    def E(self) -> float:
        """Young's modulus (Pa)"""
        return self.material_array[0, 0]
    
    @property
    def I_y(self) -> float:
        """Moment of inertia about y-axis (m⁴)"""
        return self.geometry_array[0, 3]
    
    @property
    def I_z(self) -> float:
        """Moment of inertia about z-axis (m⁴)"""
        return self.geometry_array[0, 4]
    
    @property
    def G(self) -> float:
        """Shear modulus (Pa)"""
        return self.material_array[0, 1]
    
    @property
    def J_t(self) -> float:
        """Torsional moment of inertia (m⁴)"""
        return self.geometry_array[0, 6]
    
    @property
    def warping_inertia_y(self) -> float:
        """Warping constant about y-axis (m⁶)"""
        return self.geometry_array[0, 18]

    @property  
    def warping_inertia_z(self) -> float:
        """Warping constant about z-axis (m⁶)"""
        return self.geometry_array[0, 19]

    @property
    def integration_jacobian(self) -> float:
        """Shortcut to strain operator's Jacobian"""
        return self.strain_displacement_operator.jacobian

    @property
    def integration_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gauss quadrature points/weights"""
        return np.polynomial.legendre.leggauss(self.quadrature_order)

    # Operator-compatible formulation methods ----------------------------------
    def shape_functions(self, xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shape functions and natural derivatives for Kᵉ assembly."""
        return self.shape_function_operator.natural_coordinate_form(xi)

    def B_matrix(self, dN_dξ: np.ndarray, d2N_dξ2: np.ndarray) -> np.ndarray:
        """B̃-matrix in natural coordinates (∫ B̃ᵀ D B̃ |J| dξ)."""
        return self.strain_displacement_operator.natural_coordinate_form(dN_dξ, d2N_dξ2)

    def D_matrix(self) -> np.ndarray:
        """D-matrix for stiffness assembly (4×4)."""
        return self.material_stiffness_operator.assembly_form()
    
    # Tensor computations ------------------------------------------------------
    def element_stiffness_matrix(self, job_results_dir: str = None) -> np.ndarray:
        """
        Compute stiffness matrix using operator classes with EXACTLY matching logging
        to original implementation. Maintains same debug output format while using
        the new ShapeFunctionOperator, StrainDisplacementOperator, and MaterialStiffnessOperator.
        """
        # Configure logging (identical to original)
        self.configure_element_stiffness_logging(job_results_dir)
    
        # Initialize (same variables as original)
        xi, w = self.integration_points
        detJ = self.L / 2  # For logging purposes (not used in computation)
        Ke = np.zeros((12, 12))
    
        # Get material matrix via new operator
        D = self.material_stiffness_operator.assembly_form()
    
        # Identical initial logging
        self.logger.debug(f"Element Stiffness Matrix Computation (Element ID: {self.element_id})")
        self.logger.debug(f"Element Length (L): {self.L:.6e}, Jacobian determinant (detJ): {detJ:.6e}")
        self.logger.debug("Material Stiffness Matrix (D):")
        self.logger.debug(np.array2string(D, precision=6, suppress_small=True))

        # Gauss Quadrature Integration (same loop structure)
        for g, (xi_g, w_g) in enumerate(zip(xi, w)):
            # Get shape functions via new operator (same output format)
            _, dN_dξ, d2N_dξ2 = self.shape_function_operator.natural_coordinate_form(np.array([xi_g]))
        
            # Get B-matrix via new operator (physical coordinates)
            B = self.strain_displacement_operator.physical_coordinate_form(dN_dξ, d2N_dξ2)[0]
        
            # Compute contribution (weight only - Jacobian already in B)
            Ke_contribution = B.T @ D @ B * w_g  # detJ already accounted for
            Ke += Ke_contribution

            # ----- EXACTLY matching logging format -----
            self.logger.debug(f"\n----- Gauss Point {g + 1}/{len(xi)} -----")
            self.logger.debug(f"Natural Coordinate (xi): {xi_g:.6e}, Gauss Weight: {w_g:.6e}")

            self.logger.debug("Shape Function First Derivatives (dN_dxi):")
            self.logger.debug(np.array2string(dN_dξ.squeeze(), precision=6, suppress_small=True))

            self.logger.debug("Shape Function Second Derivatives (d2N_dxi2):")
            self.logger.debug(np.array2string(d2N_dξ2.squeeze(), precision=6, suppress_small=True))

            self.logger.debug("Strain-Displacement Matrix (B):")
            self.logger.debug(np.array2string(B, precision=6, suppress_small=True))

            self.logger.debug("Strain-Displacement Matrix Transpose (B.T):")
            self.logger.debug(np.array2string(B.T, precision=6, suppress_small=True))

            self.logger.debug("Element Stiffness Matrix Contribution at Gauss Point:")
            self.logger.debug(np.array2string(Ke_contribution, precision=6, suppress_small=True))
            self.logger.debug("")  # Blank line for readability

        # Final logging (identical format)
        self.logger.debug("\n===== Final Element Stiffness Matrix (Ke) =====")
        self.logger.debug(np.array2string(Ke, precision=6, suppress_small=True))

        return Ke

    def element_force_vector(self, job_results_dir: str = None) -> np.ndarray:
        """Compute the element force vector for 3D Euler-Bernoulli beam using new operator classes.

        Mathematical Formulation
        -----------------------
        1. Distributed Loads:
        Fᵉ = ∫ Nᵀ q dx ≈ ∑ N(ξ)ᵀ q(ξ) w(ξ) |J|
        where |J| = L/2 is the Jacobian determinant

        2. Point Loads:
        Fᵉ = N(xₚ)ᵀ P
        where P = [Fx, Fy, Fz, Mx, My, Mz]

        Parameters
        ----------
        job_results_dir : str, optional
            Directory path for debug logs (matches original implementation)

        Returns
        -------
        np.ndarray
            Force vector [F₁ₓ, F₁ᵧ, F₁_z, M₁ₓ, M₁ᵧ, M₁_z, F₂ₓ, F₂ᵧ, F₂_z, M₂ₓ, M₂ᵧ, M₂_z]
        """
        # 1. INITIALIZATION (PRESERVING ORIGINAL LOGGING)
        self.configure_element_force_logging(job_results_dir)
        Fe = np.zeros(12, dtype=np.float64)
    
        self.logger.debug(f"Element Force Vector Computation (Element ID: {self.element_id})")
        self.logger.debug(f"Element Length: {self.L:.6e} m")

        # 2. DISTRIBUTED LOAD PROCESSING
        if self.distributed_load_array.size > 0:
            try:
                xi_gauss, weights = self.integration_points
                x_gauss = (xi_gauss + 1) * (self.L / 2) + self.x_start

                # Input validation
                if not np.all(np.isfinite(x_gauss)):
                    raise ValueError("Invalid Gauss point coordinates")

                # Use the enhanced interpolation operator
                interpolator = LoadInterpolationOperator(
                    distributed_loads_array=self.distributed_load_array,
                    boundary_mode="error",
                    interpolation_order="cubic",
                    n_gauss_points=self.quadrature_order
                )
                q_gauss = interpolator.interpolate(x_gauss)

                # Vectorized shape function evaluation
                N = np.stack([self.shape_function_operator.natural_coordinate_form(xi)[0][0]
                      for xi in xi_gauss])

                # Numerical stability check
                if np.any(np.abs(N) > 1e6):
                    self.logger.warning("Large shape function values detected")

                Fe_dist = np.einsum("gij,gj,g->i", N, q_gauss, weights) * (self.L / 2)
                Fe += Fe_dist

                # ORIGINAL LOGGING FORMAT
                self.logger.debug("\n===== Distributed Load Contribution =====")
                for g, (xi_g, w_g) in enumerate(zip(xi_gauss, weights)):
                    self.logger.debug(f"\n----- Gauss Point {g + 1}/{len(xi_gauss)} -----")
                    self.logger.debug(f"ξ (xi): {xi_g:.6e}, Weight: {w_g:.6e}")
                    self.logger.debug("Shape Function Values (N):")
                    self.logger.debug(np.array2string(N[g], precision=6, suppress_small=True))
                    self.logger.debug("Interpolated Load Values (q_gauss):")
                    self.logger.debug(np.array2string(q_gauss[g], precision=6, suppress_small=True))
                    self.logger.debug("Force Contribution:")
                    self.logger.debug(np.array2string(Fe_dist.reshape(1, -1), precision=6, suppress_small=True))

            except Exception as e:
                self.logger.error(f"Distributed load processing failed: {str(e)}")
                raise

        # 3. POINT LOAD PROCESSING
        if self.point_load_array.size > 0:
            for load_idx, load in enumerate(self.point_load_array):
                try:
                    x_p = float(load[0])
                    F_p = load[3:9].astype(np.float64)

                    # Robust boundary check with tolerance
                    tol = 1e-12 * self.L
                    in_element = (self.x_start - tol <= x_p <= self.x_end + tol) if np.isclose(self.x_end, self.x_global_end) \
                                else (self.x_start - tol <= x_p < self.x_end + tol)
                    if not in_element:
                        continue

                    xi_p = 2 * (x_p - self.x_start) / self.L - 1
                    N_p = self.shape_function_operator.natural_coordinate_form(np.array([xi_p]))[0][0]

                    # Physical units check
                    if np.max(np.abs(F_p[:3])) > 1e12:  # > 1 TN force check
                        self.logger.warning(f"Extreme force magnitude at load {load_idx}")

                    Fe_trans = np.einsum("ij,j->i", N_p[[0,1,2,6,7,8], :3], F_p[:3])
                    Fe_rot = np.einsum("ij,j->i", N_p[[3,4,5,9,10,11], 3:], F_p[3:])

                    Fe[[0,1,2,6,7,8]] += Fe_trans
                    Fe[[3,4,5,9,10,11]] += Fe_rot

                    # ORIGINAL POINT LOAD LOGGING
                    self.logger.debug(f"\nPoint Load at x_p = {x_p:.6e} (xi_p = {xi_p:.6e})")
                    self.logger.debug(f"Load Vector: {F_p}")
                    self.logger.debug("Shape Functions:")
                    self.logger.debug(np.array2string(N_p, precision=6, suppress_small=True))
                    self.logger.debug("Translational Contribution:")
                    self.logger.debug(np.array2string(Fe_trans.reshape(1, -1), precision=6, suppress_small=True))
                    self.logger.debug("Rotational Contribution:")
                    self.logger.debug(np.array2string(Fe_rot.reshape(1, -1), precision=6, suppress_small=True))

                except Exception as e:
                    self.logger.error(f"Point load {load_idx} processing failed: {str(e)}")
                    continue

        # 4. FINAL VALIDATION AND OUTPUT
        # NaN/Inf check
        if not np.all(np.isfinite(Fe)):
            raise ValueError("Non-finite values in force vector")
    
        # Unit consistency check
        if np.max(np.abs(Fe[:3])) > 1e12 or np.max(np.abs(Fe[3:6])) > 1e12:
            self.logger.warning("Extreme force/moment values detected")
    
        # Expected magnitude check
        expected_scale = max(1.0, np.max(np.abs(self.distributed_load_array[:,3:9])) * self.L) if self.distributed_load_array.size > 0 else 1.0
        if np.max(np.abs(Fe)) > 1e6 * expected_scale:
            self.logger.warning("Force vector magnitude exceeds expected scale")

        self.logger.debug("\n===== Final Element Force Vector =====")
        self.logger.debug(np.array2string(Fe.reshape(1, -1), precision=6, suppress_small=True))

        return Fe

    # Additional functionality -------------------------------------------------
    @property
    def connectivity(self) -> np.ndarray:
        """Element node connectivity"""
        return self.mesh_dictionary["connectivity"][self.element_id]

    def __repr__(self) -> str:
        return (f"EulerBernoulliBeamElement6DOF(element_id={self.element_id}, "
                f"L={self.L:.2e}m, E={self.E:.2e}Pa, "
                f"quad_order={self.quadrature_order})")
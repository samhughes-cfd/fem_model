# pre_processing\element_library\euler_bernoulli\euler_bernoulli_6DOF.py

import numpy as np
from typing import Tuple
import os

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
    - Integrated logging system for stiffness matrices and force vectors
    """
    
    def __init__(self, 
                 geometry_array: np.ndarray,
                 material_array: np.ndarray,
                 mesh_dictionary: dict,
                 point_load_array: np.ndarray,
                 distributed_load_array: np.ndarray,
                 element_id: int,
                 job_results_dir: str,
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
            logger_operator: Configured logger operator for element-specific logging
        """

        element_id = int(element_id)

        super().__init__(
            geometry_array=geometry_array,
            material_array=material_array,
            mesh_dictionary=mesh_dictionary,
            point_load_array=point_load_array,
            distributed_load_array=distributed_load_array,
            dof_per_node=6,
            element_id=element_id,
            job_results_dir=job_results_dir
        )
        
        self.quadrature_order = quadrature_order

        # Initialize element properties and validate
        self._init_element_geometry()
        self._validate_element_properties()
        self._validate_logging_setup()
    
        # Initialize operator classes
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
        
    def _validate_logging_setup(self):
        """Ensure logging infrastructure is operational"""
        if not self.logger_operator:
            raise RuntimeError(f"Element {self.element_id} has no logger operator")
            
        # Verify log file paths
        required_categories = ["stiffness", "force"]
        for cat in required_categories:
            log_path = self.logger_operator._get_log_path(cat)
            if not os.path.exists(os.path.dirname(log_path)):
                raise FileNotFoundError(f"Missing directory for {cat} logs")
                
            if not os.access(os.path.dirname(log_path), os.W_OK):
                raise PermissionError(f"Cannot write to {cat} log directory")

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
    
    # Ke tensor computations ---------------------------------------------------------
    def element_stiffness_matrix(self) -> np.ndarray:
        """
        Compute stiffness matrix using operator classes with integrated logging
        """
        self._validate_logging_setup()

        Ke = np.zeros((12, 12), dtype=np.float64)
        D = self.material_stiffness_operator.assembly_form()
        xi, w = self.integration_points

        if self.logger_operator:  # Modified logging block
            self.logger_operator.log_text("stiffness", f"\n=== Element {self.element_id} Stiffness Matrix Computation ===")
            self.logger_operator.log_matrix("stiffness", D, {"name": "Material Stiffness Matrix (D)"})
        
        for g, (xi_g, w_g) in enumerate(zip(xi, w)):
            _, dN_dξ, d2N_dξ2 = self.shape_function_operator.natural_coordinate_form(np.array([xi_g]))
            B = self.strain_displacement_operator.physical_coordinate_form(dN_dξ, d2N_dξ2)[0]
            Ke_contribution = B.T @ D @ B * w_g
            Ke += Ke_contribution

            if self.logger_operator:
                self._log_gauss_point_stiffness(g, xi_g, w_g, dN_dξ, d2N_dξ2, B, Ke_contribution)

        if self.logger_operator:  # Modified logging block
            self.logger_operator.log_matrix("stiffness", Ke, {"name": "Final Element Stiffness Matrix"})
            self.logger_operator.flush("stiffness")

        return Ke

    # Fe tensor computations ---------------------------------------------------------
    def element_force_vector(self) -> np.ndarray:
        """Compute the element force vector including distributed and point loads.

        Returns
        -------
        np.ndarray [12]
            Force vector in local coordinates ordered as:
            [Fx1, Fy1, Fz1, Mx1, My1, Mz1, Fx2, Fy2, Fz2, Mx2, My2, Mz2]

        Notes
        -----
        Combines contributions from:
        - Distributed loads: F_dist = ∫ N^T q dx
        - Point loads: F_point = N(x_p)^T P
        """
        self._validate_logging_setup()

        Fe = np.zeros(12, dtype=np.float64)
        
        if self.logger_operator:  # Modified logging call
            self.logger_operator.log_text("force", f"\n=== Element {self.element_id} Force Vector Computation ===")

        # Process distributed loads
        if self.distributed_load_array.size > 0:
            Fe += self._compute_distributed_load_contribution()

        # Process point loads
        if self.point_load_array.size > 0:
            Fe += self._compute_point_load_contribution()

        if self.logger_operator:  # Modified logging block
            self.logger_operator.log_matrix("force", Fe.reshape(1, -1), {"name": "Final Force Vector"})
            self.logger_operator.flush("force")

        return Fe

    # Fe - distributed load contribution ------------------------------------------------
    def _compute_distributed_load_contribution(self) -> np.ndarray:
        """Compute distributed load contribution using Gauss quadrature."""
        xi_gauss, weights = self.integration_points
        x_gauss = (xi_gauss + 1) * (self.L / 2) + self.x_start
        Fe_dist = np.zeros(12, dtype=np.float64)

        try:
            interpolator = LoadInterpolationOperator(
                distributed_loads_array=self.distributed_load_array,
                boundary_mode="error",
                interpolation_order="cubic",
                n_gauss_points=self.quadrature_order
            )
            q_gauss = interpolator.interpolate(x_gauss)
            N = np.stack([self.shape_function_operator.natural_coordinate_form(xi)[0][0]
                        for xi in xi_gauss])
            Fe_dist = np.einsum("gij,gj,g->i", N, q_gauss, weights) * (self.L / 2)

            if self.logger_operator:
                self._log_distributed_loads(xi_gauss, weights, N, q_gauss, Fe_dist)

        except Exception as e:
            self.logger.error(f"Distributed load error: {str(e)}")
            raise

        return Fe_dist
    
    # Fe - point load contribution
    def _compute_point_load_contribution(self) -> np.ndarray:
        """Compute point load contributions using shape function evaluation."""
        Fe_point = np.zeros(12, dtype=np.float64)
        
        for load in self.point_load_array:
            x_p = float(load[0])
            F_p = load[3:9].astype(np.float64)
            
            if not self._is_point_in_element(x_p):
                continue

            xi_p = 2 * (x_p - self.x_start) / self.L - 1
            N_p = self.shape_function_operator.natural_coordinate_form(np.array([xi_p]))[0][0]

            Fe_trans = np.einsum("ij,j->i", N_p[[0,1,2,6,7,8], :3], F_p[:3])
            Fe_rot = np.einsum("ij,j->i", N_p[[3,4,5,9,10,11], 3:], F_p[3:])
            Fe_point[[0,1,2,6,7,8]] += Fe_trans
            Fe_point[[3,4,5,9,10,11]] += Fe_rot

            if self.logger_operator:
                self._log_point_load(x_p, xi_p, F_p, N_p, Fe_trans, Fe_rot)
        
        return Fe_point

    # Stiffness logging helpers ----------------------------------------------------------
    def _log_gauss_point_stiffness(self, gp_idx: int, xi: float, weight: float,
                                  B: np.ndarray, contribution: np.ndarray):
        """Log detailed stiffness matrix integration data."""
        metadata = {
            "name": f"Gauss Point {gp_idx+1}",
            "precision": 6,
            "max_line_width": 120
        }
        
        self.log_text("stiffness", f"\nGP {gp_idx+1}/{self.quadrature_order}: ξ={xi:.4f}, w={weight:.4f}")
        self.log_matrix("stiffness", B, {**metadata, "name": "B-Matrix"})
        self.log_matrix("stiffness", contribution, {**metadata, "name": "Contribution"})

    # Force logging helpers ----------------------------------------------------------
    def _log_distributed_loads(self, xi: np.ndarray, weights: np.ndarray,
                              N: np.ndarray, q: np.ndarray, Fe: np.ndarray):
        """Log distributed load integration details."""
        if self.logger_operator:  # Modified logging calls
            self.logger_operator.log_text("force", "\nDistributed Load Contributions:")
            for gp, (xi_g, w_g) in enumerate(zip(xi, weights)):
                self.logger_operator.log_text("force", f"\nGP {gp+1}: ξ={xi_g:.4f}, w={w_g:.4f}")
                self.logger_operator.log_matrix("force", N[gp], {"name": "Shape Functions", "precision": 4})
                self.logger_operator.log_matrix("force", q[gp], {"name": "Load Vector", "precision": 4})
            self.logger_operator.log_matrix("force", Fe, {"name": "Total Contribution", "precision": 6})

    def _log_point_load(self, x: float, xi: float, F: np.ndarray,
                       N: np.ndarray, trans: np.ndarray, rot: np.ndarray):
        """Log point load application details."""
        if self.logger_operator:  # Modified logging calls
            self.logger_operator.log_text("force", f"\nPoint Load @ x={x:.4f} (ξ={xi:.4f})")
            self.logger_operator.log_matrix("force", F, {"name": "Global Load Vector", "precision": 6})
            self.logger_operator.log_matrix("force", N, {"name": "Shape Functions", "precision": 4})
            self.logger_operator.log_matrix("force", trans, {"name": "Translational Contribution", "precision": 6})
            self.logger_operator.log_matrix("force", rot, {"name": "Rotational Contribution", "precision": 6})

    # Utility methods ----------------------------------------------------------
    def _is_point_in_element(self, x: float) -> bool:
        """Check if point x is within element bounds with tolerance."""
        tol = 1e-12 * self.L
        if np.isclose(self.x_end, self.x_global_end):
            return (self.x_start - tol <= x <= self.x_end + tol)
        return (self.x_start - tol <= x < self.x_end + tol)

    def __repr__(self) -> str:
        return (f"EulerBernoulliBeam6DOF(id={self.element_id}, L={self.L:.2e}m, "
                f"E={self.E:.1e}Pa, quad={self.quadrature_order})")
import numpy as np
import pre_processing.element_library.element_1D_base
from pre_processing.element_library.utilities.dof_mapping import expand_dof_mapping
from pre_processing.element_library.utilities.shape_function_library.timoshenko_sf import timoshenko_shape_functions
from pre_processing.element_library.utilities.gauss_quadrature import integrate_matrix, integrate_vector
from pre_processing.element_library.utilities.coordinate_transform import natural_to_physical
from pre_processing.element_library.utilities.interpolate_loads import interpolate_loads

class TimoshenkoBeamElement3DOF(pre_processing.element_library.element_1D_base.Element1DBase):
    """
    Timoshenko beam element (1D) with 3 DOFs per node:
    (u_x, u_y, θ_z), where θ_z accounts for both bending and shear effects.
    Expands to a full 12 DOFs per element system (12x12 stiffness matrix and 12x1 force vector).
    """

    def __init__(self, element_id: int, material_array: np.ndarray, geometry_array: np.ndarray, 
                 mesh_data: dict, node_positions: np.ndarray, loads_array: np.ndarray):
        """
        Initializes the Timoshenko beam element with material properties, geometry,
        and associated mesh data.

        Args:
            element_id (int): Unique identifier for the element.
            material_array (np.ndarray): Material properties containing Young's modulus and shear modulus.
            geometry_array (np.ndarray): Geometry properties containing cross-sectional area and moment of inertia.
            mesh_data (dict): Contains nodal connectivity and global coordinates.
            node_positions (np.ndarray): Nodal positions in global coordinates.
            loads_array (np.ndarray): External force and moment distribution.
        """
        super().__init__(
            geometry_array=geometry_array,
            material_array=material_array,
            mesh_dictionary=mesh_data,
            load_array=loads_array,
            dof_per_node=3
        )

        # Store precomputed Jacobian determinant for coordinate transformations
        self.detJ = self._jacobians[element_id]["jacobian_determinant"]

    def shape_functions(self, xi: float) -> tuple:
        """
        Evaluates the shape functions and their derivatives for the Timoshenko beam element
        at a given natural coordinate ξ.

        Args:
            xi (float): Natural coordinate (ξ) in range [-1, 1].

        Returns:
            tuple: (N, dN_dxi, d2N_dxi2)
                - `N` (ndarray): Shape function vector.
                - `dN_dxi` (ndarray): First derivative of shape functions.
                - `d2N_dxi2` (ndarray): Second derivative of shape functions.
        """
        element_index = self.get_element_index()
        element_length = self.element_lengths_array[element_index]
        return timoshenko_shape_functions(xi, element_length)

    def strain_displacement_matrix(self, dN_dxi: np.ndarray) -> tuple:
        """
        Constructs the strain-displacement matrix `B`, which relates nodal displacements to element strains.

        Args:
            dN_dxi (np.ndarray): First derivative of shape functions w.r.t. natural coordinate ξ.

        Returns:
            tuple: (B, B_axial, B_bending, B_shear)
                - `B` (ndarray): Combined strain-displacement matrix (3x3).
                - `B_axial` (ndarray): Contribution to axial strain.
                - `B_bending` (ndarray): Contribution to bending strain.
                - `B_shear` (ndarray): Contribution to shear strain.
        """
        dxi_dx = 1.0 / self.detJ

        B_axial = np.zeros(3)
        B_axial[0] = dN_dxi[0] * dxi_dx
        B_axial[2] = dN_dxi[2] * dxi_dx  # θ_z contributes to axial strain

        B_bending = np.zeros(3)
        B_bending[1] = dN_dxi[1] * (dxi_dx**2)  # Pure bending
        B_bending[2] = dN_dxi[2] * dxi_dx       # Shear contribution included in θ_z

        B_shear = np.zeros(3)
        B_shear[1] = dN_dxi[1] * dxi_dx         # Shear force contributes to θ_z
        B_shear[2] = dN_dxi[2] * dxi_dx         # Shear strain included in θ_z

        B = np.vstack([B_axial, B_bending, B_shear])
        return B, B_axial, B_bending, B_shear

    def material_stiffness_matrix(self) -> tuple:
        """
        Constructs the element material stiffness (constitutive) matrix `D`, representing
        axial, bending, and shear stiffness.

        Returns:
            tuple: (D, D_axial, D_bending, D_shear)
                - `D` (ndarray): Full material stiffness matrix (3x3).
                - `D_axial` (float): Axial stiffness.
                - `D_bending` (float): Bending stiffness.
                - `D_shear` (float): Shear stiffness.
        """
        D_axial = self.E * self.A
        D_bending = self.E * self.I_z
        D_shear = self.G * self.ks * self.A  # Shear stiffness applied to θ_z

        D = np.diag([D_axial, D_bending + D_shear, D_shear])  # θ_z stiffness includes shear
        return D, D_axial, D_bending, D_shear

    def element_stiffness_matrix(self):
        """
        Computes the element stiffness matrix via numerical integration (Gauss quadrature)
        and expands it to a 12x12 system.
        """
        D, D_axial, D_bending, D_shear = self.material_stiffness_matrix()

        def integrand_stiffness_matrix(xi: np.ndarray, D_component: float, component: int) -> np.ndarray:
            """
            Computes stiffness contributions per component (axial, bending, shear).
            """
            _, B_axial, B_bending, B_shear = self.strain_displacement_matrix(self.shape_functions(xi[0])[1])
            B_component = [B_axial, B_bending, B_shear][component]
            return B_component.T @ D_component @ B_component

        Ke_axial = integrate_matrix(3, lambda xi: integrand_stiffness_matrix(xi, D_axial, 0) * self.detJ, dim=1)
        Ke_bending = integrate_matrix(3, lambda xi: integrand_stiffness_matrix(xi, D_bending, 1) * self.detJ, dim=1)
        Ke_shear = integrate_matrix(2, lambda xi: integrand_stiffness_matrix(xi, D_shear, 2) * self.detJ, dim=1)

        Ke_reduced = Ke_axial + Ke_bending + Ke_shear

        dof_map = [0, 1, 5, 6, 7, 11]  # Axial, transverse, and θ_z DOFs per node
        self.Ke = expand_dof_mapping(Ke_reduced, full_size=12, dof_map=dof_map)
        self.validate_matrices()

    def element_force_vector(self):
        """
        Computes the element force vector via numerical integration (Gauss quadrature)
        and expands it to a 12x1 system.
        """
        def integrand_force_vector(xi: np.ndarray) -> np.ndarray:
            """
            Computes the force contributions.
            """
            shape_data = self.shape_functions(xi[0])
            N, _, _ = shape_data
            x_phys = natural_to_physical(xi.reshape(-1, 1), self.get_element_index(), self.mesh_dictionary, self.element_lengths_array)[:, 0]
            q_xi = interpolate_loads(x_phys[0], self.load_array)
            return N.T @ q_xi

        Fe_reduced = integrate_vector(3, lambda xi: integrand_force_vector(xi) * self.detJ, dim=1)

        dof_map = [0, 1, 5, 6, 7, 11]  # u_x, u_y, θ_z (with shear)
        self.Fe = expand_dof_mapping(Fe_reduced, full_size=12, dof_map=dof_map)
        self.validate_matrices()

    def validate_matrices(self):
        """ Validates that Ke and Fe have the correct dimensions. """
        assert self.Ke.shape == (12, 12), f"Ke shape mismatch: Expected (12,12), got {self.Ke.shape}"
        assert self.Fe.shape == (12,), f"Fe shape mismatch: Expected (12,), got {self.Fe.shape}"
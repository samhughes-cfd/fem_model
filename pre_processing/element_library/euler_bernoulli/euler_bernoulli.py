# pre_processing/element_library/euler_bernoulli/euler_bernoulli_beam_element.py

import numpy as np
from pre_processing.element_library.utilities.dof_mapping import expand_dof_mapping
from pre_processing.element_library.utilities.shape_function_library.euler_bernoulli_sf import euler_bernoulli_shape_functions
from pre_processing.element_library.utilities.gauss_quadrature import get_gauss_points
from pre_processing.element_library.utilities.coordinate_transform import natural_to_physical
from pre_processing.element_library.utilities.interpolate_loads import interpolate_loads


class EulerBernoulliBeamElement:
    """
    Euler-Bernoulli beam element (1D) with 6 DOFs per node:
    (u_x, u_y, u_z, θ_x, θ_y, θ_z).
    Expands to a full 12 DOFs per element system (12x12 stiffness matrix and 12x1 force vector).
    """

    # Class-level indices for geometry and material arrays
    GEOMETRY_A_INDEX = 1    # Cross-sectional area (A)
    GEOMETRY_IZ_INDEX = 3   # Moment of inertia about the z-axis (Iz)
    MATERIAL_E_INDEX = 0    # Young's Modulus (E)

    def __init__(self, element_id: int, material_array: np.ndarray, geometry_array: np.ndarray,
                 mesh_data: dict, load_array: np.ndarray):
        """
        Initializes the Euler-Bernoulli beam element with material properties, geometry,
        and associated mesh data.

        - `element_id` (int): Unique identifier for the element.
        - `material_array` (np.ndarray): Material properties containing Young's modulus.
        - `geometry_array` (np.ndarray): Geometry properties containing cross-sectional area and Iz.
        - `mesh_data` (dict): Contains nodal connectivity and global coordinates.
        - `load_array` (np.ndarray): External force and moment distribution.
        """

        from pre_processing.element_library.element_1D_base import Element1DBase

        # Extract material and geometric properties
        self.A = geometry_array[0, self.GEOMETRY_A_INDEX]
        self.I_z = geometry_array[0, self.GEOMETRY_IZ_INDEX]
        self.E = material_array[0, self.MATERIAL_E_INDEX]

        # Initialize the base class with 6 DOFs per node
        super().__init__(geometry_array, material_array, mesh_data, load_array, dof_per_node=6)

        # Store precomputed Jacobian determinant for coordinate transformations
        self.detJ = self._jacobians[element_id]["jacobian_determinant"]

    def shape_functions(self, xi: float) -> tuple:
        """
        Evaluates the shape functions and their derivatives for the Euler-Bernoulli beam element
        at a given natural coordinate ξ.

        - `xi` (float): Natural coordinate (ξ) in range [-1, 1].
        - Returns: Tuple containing (N, dN_dxi, ddN_dxi) shape functions.
        """
        element_index = self.get_element_index()
        element_length = self.element_lengths_array[element_index]
        return euler_bernoulli_shape_functions(xi, element_length)

    def material_stiffness_matrix(self) -> np.ndarray:
        """
        Constructs the element material stiffness (constitutive) matrix `D`, representing
        axial and bending stiffness.

        - Returns: (2x2) diagonal matrix.
        """
        return np.diag([self.E * self.A, self.E * self.I_z])  # Shape: (2,2)

    def strain_displacement_matrix(self, dN_dxi: np.ndarray) -> np.ndarray:
        """
        Constructs the strain-displacement matrix `B`, which relates nodal displacements
        to element strains.

        - `dN_dxi` (np.ndarray): Derivative of shape functions w.r.t. natural coordinate ξ.
        - Returns: (2x6) strain-displacement matrix.
        """
        dxi_dx = 1.0 / self.detJ  # Compute inverse of Jacobian determinant for transformation

        # Initialize axial and bending strain-displacement matrices
        B_axial = np.zeros(6)
        B_axial[0] = dN_dxi[0] * dxi_dx  # Axial force at start node
        B_axial[3] = dN_dxi[3] * dxi_dx  # Axial force at end node

        B_bending = np.zeros(6)
        B_bending[1] = dN_dxi[1] * (dxi_dx ** 2)  # Shear force at start
        B_bending[2] = dN_dxi[2] * (dxi_dx ** 2)  # Bending moment at start
        B_bending[4] = dN_dxi[4] * (dxi_dx ** 2)  # Shear force at end
        B_bending[5] = dN_dxi[5] * (dxi_dx ** 2)  # Bending moment at end

        # Assemble final strain-displacement matrix (2x6)
        return np.vstack([B_axial, B_bending])

    def element_stiffness_matrix(self):
        """
        Computes the element stiffness matrix via numerical integration (Gauss quadrature)
        and expands it to a 12x12 system.
        """
        gauss_points, weights = get_gauss_points(n=3, dim=1)
        shape_data = {xi[0]: self.shape_functions(xi[0]) for xi in gauss_points}
        D = self.material_stiffness_matrix()

        def integrand_stiffness_matrix(xi_index: int) -> np.ndarray:
            xi = gauss_points[xi_index][0]
            xi_closest = min(shape_data.keys(), key=lambda k: abs(k - xi))
            _, dN_dxi, _ = shape_data[xi_closest]
            B = self.strain_displacement_matrix(dN_dxi)
            return B.T @ D @ B

        Ke_reduced = np.sum(
            [weights[i] * integrand_stiffness_matrix(i) * self.detJ for i in range(len(weights))],
            axis=0
        )

        assert Ke_reduced.shape == (6, 6), "Ke shape mismatch: expected (6,6)"

        dof_map = [0, 1, 5, 6, 7, 11]
        self.Ke = expand_dof_mapping(Ke_reduced, full_size=12, dof_map=dof_map)

    def element_force_vector(self):
        """
        Computes the element force vector via numerical integration (Gauss quadrature)
        and expands it to a 12x1 system.
        """
        gauss_points, weights = get_gauss_points(n=3, dim=1)
        xi_array = np.array([xi[0] for xi in gauss_points]).reshape(-1, 1)
        element_index = self.get_element_index()
        x_phys_array = natural_to_physical(xi_array, element_index, self.mesh_data, self.element_lengths_array)[:, 0]
        shape_data = {xi[0]: self.shape_functions(xi[0]) for xi in gauss_points}
        q_xi_array = np.array([interpolate_loads(x, self.load_array) for x in x_phys_array])

        def integrand_force_vector(xi_index: int) -> np.ndarray:
            xi = xi_array[xi_index, 0]
            xi_closest = min(shape_data.keys(), key=lambda k: abs(k - xi))
            N, _, _ = shape_data[xi_closest]
            return N.T @ q_xi_array[xi_index]

        Fe_reduced = np.sum(
            [weights[i] * integrand_force_vector(i) * self.detJ for i in range(len(weights))],
            axis=0
        )

        assert Fe_reduced.shape == (6,), "Fe shape mismatch: expected (6,)"

        dof_map = [0, 1, 5, 6, 7, 11]
        self.Fe = expand_dof_mapping(Fe_reduced, full_size=12, dof_map=dof_map)

    def validate_matrices(self):
        """ Validates that Ke and Fe have the correct dimensions. """
        assert self.Ke.shape == (12, 12), f"Ke shape mismatch: Expected (12,12), got {self.Ke.shape}"
        assert self.Fe.shape == (12,), f"Fe shape mismatch: Expected (12,), got {self.Fe.shape}"
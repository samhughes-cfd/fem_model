import numpy as np
from pre_processing.element_library.element_1D_base import Element1DBase
from pre_processing.element_library.utilities.gauss_quadrature import integrate_matrix
from pre_processing.element_library.utilities.shape_function_library.timoshenko_sf import timoshenko_shape_functions


class TimoshenkoBeamElement(Element1DBase):
    """Timoshenko beam element (3D) with 3 DOFs per node: (u_x, u_y, θ_z)."""

    def __init__(self, element_id, geometry, material, section_props):
        """
        Initializes a Timoshenko beam element.

        Args:
            element_id (int): Unique identifier for this element.
            geometry: Geometry object (must implement get_element_length).
            material: Material object with E (Young's modulus) and G (Shear modulus).
            section_props (dict): 3D section properties including:
                - 'A': Cross-sectional area
                - 'Iz': Second moment of area about the z-axis
                - 'ks': Shear correction factor
        """
        self.A = section_props["A"]
        self.I_bending = section_props["Iz"]
        self.ks = section_props["ks"]  # Shear correction factor
        self.G = material["G"]

        super().__init__(element_id, material, section_props, geometry, dof_per_node=3, dof_map=[0, 1, 5])

    def element_stiffness_matrix(self):
        """
        Computes the Timoshenko beam stiffness matrix, incorporating shear flexibility.
        """
        n_gauss = 3
        Ke_reduced = integrate_matrix(n_gauss, self.B_transpose_D_B_timoshenko, self.jacobian_func, dim=1)

        self.Ke = np.zeros((12, 12))  # Full 6DOF per node structure
        dof_map = self.dof_map + [d + 6 for d in self.dof_map]  # Map both nodes

        for i in range(6):
            for j in range(6):
                self.Ke[dof_map[i], dof_map[j]] = Ke_reduced[i, j]

    def B_transpose_D_B_timoshenko(self, xi):
        """
        Integrand for K_e: (B^T)(D)(B) at a given natural coordinate.

        Args:
            xi (np.ndarray): 1D array with a single float in [-1, 1].

        Returns:
            np.ndarray: 6×6 matrix representing B^T * D * B at xi.
        """
        xi_scalar = xi[0]
        B = self.strain_displacement_matrix(xi_scalar)
        D = self.material_stiffness_matrix()
        return B.T @ D @ B

    def strain_displacement_matrix(self, xi):
        """
        Builds the strain-displacement matrix (3×6) for axial, bending, and shear strains.

        Returns:
            np.ndarray: A 3×6 matrix with:
                - **Row 1**: Axial strain (`du/dx`)
                - **Row 2**: Bending curvature (`d²w/dx²`)
                - **Row 3**: Shear strain (`dθ/dx`)
        """
        N, dN_dxi, d2N_dxi2 = self.shape_functions(xi)
        L = self.geometry.get_element_length(self.element_id)
        dxi_dx = 2.0 / L

        # Axial strain row (epsilon = du/dx)
        B_axial = np.zeros(6)
        B_axial[0] = dN_dxi[0] * dxi_dx
        B_axial[3] = dN_dxi[3] * dxi_dx

        # Bending strain row (kappa = d²w/dx²)
        d2N_dx2 = d2N_dxi2 * (dxi_dx**2)
        B_bending = np.zeros(6)
        B_bending[1] = d2N_dx2[1]
        B_bending[2] = d2N_dx2[2]
        B_bending[4] = d2N_dx2[4]
        B_bending[5] = d2N_dx2[5]

        # Shear strain row (gamma = dθ/dx)
        B_shear = np.zeros(6)
        B_shear[1] = dN_dxi[1] * dxi_dx
        B_shear[4] = dN_dxi[4] * dxi_dx

        return np.vstack([B_axial, B_bending, B_shear])

    def material_stiffness_matrix(self):
        """
        Returns the 3×3 material matrix D = diag(EA, EI, G*k*A).
        """
        E = self.material["E"]
        G = self.material["G"]
        return np.array([
            [E * self.A, 0.0, 0.0],  
            [0.0, E * self.I_bending, 0.0],
            [0.0, 0.0, G * self.ks * self.A]  
        ])

    def shape_functions(self, xi):
        """
        Retrieves Hermite polynomials (bending) and linear polynomials (axial)
        for a 2-node Timoshenko beam.

        Returns:
            tuple: (N, dN_dxi, d²N_dxi²)
        """
        L = self.geometry.get_element_length(self.element_id)
        return conv_t_shape_functions(xi, L)

    def jacobian_func(self, xi):
        """
        Computes Jacobian determinant for a linear 1D element: `L/2`.

        Returns:
            float: (L / 2.0)
        """
        L = self.geometry.get_element_length(self.element_id)
        return L / 2.0
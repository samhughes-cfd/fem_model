import numpy as np
import pre_processing.element_library.element_1D_base
from pre_processing.element_library.utilities.dof_mapping import expand_dof_mapping
from pre_processing.element_library.utilities.shape_function_library.timoshenko_sf import timoshenko_shape_functions
from pre_processing.element_library.utilities.jacobian import compute_jacobian_matrix, compute_jacobian_determinant
from pre_processing.element_library.utilities.gauss_quadrature import integrate_matrix

class TimoshenkoBeamElement(pre_processing.element_library.element_1D_base.Element1DBase):
    """
    Timoshenko beam element (1D) with 6 DOFs per node: 
    (u_x, u_y, u_z, θ_x, θ_y, θ_z).
    Expands to a full 12 DOFs per element system (12x12 stiffness matrix and 12x1 force vector).

    Attributes:
        A (float): Cross-sectional area.
        I_z (float): Second moment of area about the z-axis.
        E (float): Young's Modulus.
        G (float): Shear Modulus.
        ks (float): Shear correction factor.
    """

    # Define class-level constants for geometry and material array indices
    GEOMETRY_A_INDEX = 1    # Cross-sectional area (A)
    GEOMETRY_IZ_INDEX = 3   # Moment of inertia about the z-axis (Iz)
    #GEOMETRY_KS_INDEX = 4   # Shear correction factor (ks)
    MATERIAL_E_INDEX = 0    # Young's Modulus (E)
    MATERIAL_G_INDEX = 1    # Shear Modulus (G)

    def __init__(self, element_id: int, material_array: np.ndarray, geometry_array: np.ndarray, 
                 mesh_data: dict, node_positions: np.ndarray, loads_array: np.ndarray):
        """
        Initializes the Timoshenko beam element.

        Args:
            element_id (int): Unique identifier for this element.
            material_array (np.ndarray): Material properties array of shape (1, 4).
            geometry_array (np.ndarray): Geometry properties array of shape (1, 20).
            mesh_data (dict): Mesh data dictionary containing connectivity, element lengths, and element IDs.
            node_positions (np.ndarray): Array of node positions. Shape: (num_nodes, 3)
            loads_array (np.ndarray): Global loads array. Shape: (N, 9)
        """
        # Extract geometry and material properties using predefined indices
        self.A = geometry_array[0, self.GEOMETRY_A_INDEX]       # Cross-sectional area
        self.I_z = geometry_array[0, self.GEOMETRY_IZ_INDEX]   # Moment of inertia
        self.ks = 0.6 # for rectanguylar section (look into c-section)  #geometry_array[0, self.GEOMETRY_KS_INDEX]    # Shear correction factor
        self.E = material_array[0, self.MATERIAL_E_INDEX]      # Young's Modulus
        self.G = material_array[0, self.MATERIAL_G_INDEX]      # Shear Modulus

        # Initialize the base class
        super().__init__(
            geometry_array=geometry_array,
            material_array=material_array,
            mesh_dictionary=mesh_data,
            loads_array=loads_array,
            dof_per_node=6
        )

    def shape_functions(self, xi: float) -> tuple:
        """
        Computes the shape functions and their derivatives for the Timoshenko beam element.

        Args:
            xi (float): Natural coordinate in [-1, 1].

        Returns:
            tuple: (N, dN_dxi, d2N_dxi2)
                N (ndarray): Shape function vector (6,)
                dN_dxi (ndarray): First derivatives of shape functions w.r.t. xi (6,)
                d2N_dxi2 (ndarray): Second derivatives of shape functions w.r.t. xi (6,)
        """
        element_index = self.get_element_index()
        element_length = self.element_lengths_array[element_index]
        return timoshenko_shape_functions(xi, element_length)

    def material_stiffness_matrix(self) -> np.ndarray:
        """
        Constructs the material stiffness (constitutive) matrix D for the Timoshenko beam element.

        Returns:
            ndarray: Constitutive matrix D. Shape: (3,3)
        """
        E = self.E
        G = self.G
        A = self.A
        I_z = self.I_z
        ks = self.ks

        # Constitutive matrix for axial (Fx), bending (Mz), and shear
        D = np.diag([
            E * A,           # Axial stiffness (Fx)
            E * I_z,         # Bending stiffness (Mz)
            G * ks * A       # Shear stiffness
        ])  # Shape: (3,3)

        return D

    def strain_displacement_matrix(self, dN_dxi: np.ndarray) -> np.ndarray:
        """
        Constructs the strain-displacement matrix (B matrix) for the Timoshenko beam element.

        Args:
            dN_dxi (ndarray): First derivatives of shape functions w.r.t. xi. Shape: (6,)

        Returns:
            ndarray: Strain-displacement matrix. Shape: (3,6)
        """
        element_index = self.get_element_index()
        node_indices = self.connectivity_array[element_index]
        node_coords = self.node_coordinates_array[node_indices - 1]

        jacobian_matrix = compute_jacobian_matrix(dN_dxi.reshape(-1, 1), node_coords.reshape(-1, 1))
        detJ = compute_jacobian_determinant(jacobian_matrix)

        if detJ <= 0:
            raise ValueError(f"Invalid Jacobian determinant ({detJ}) for Element ID {self.element_id}. Check node ordering.")

        dxi_dx = 1.0 / detJ

        # Initialize strain-displacement matrices
        B_axial = np.zeros(6)
        B_axial[0] = dN_dxi[0] * dxi_dx
        B_axial[3] = dN_dxi[3] * dxi_dx

        B_bending = np.zeros(6)
        B_bending[1] = dN_dxi[1] * (dxi_dx**2)
        B_bending[4] = dN_dxi[4] * (dxi_dx**2)

        B_shear = np.zeros(6)
        B_shear[2] = dN_dxi[2] * dxi_dx
        B_shear[5] = dN_dxi[5] * dxi_dx

        return np.vstack([B_axial, B_bending, B_shear])  # Shape: (3,6)

    def element_stiffness_matrix(self):
        """
        Computes the element stiffness matrix and expands it to a 12x12 system.
        """
        def integrand_stiffness_matrix(xi: float) -> np.ndarray:
            N, dN_dxi, _ = self.shape_functions(xi)
            B = self.strain_displacement_matrix(dN_dxi)
            D = self.material_stiffness_matrix()
            return B.T @ D @ B  # Shape: (6,6)

        Ke_reduced = integrate_matrix(
            n_gauss=3,
            integrand_func=integrand_stiffness_matrix,
            jacobian_func=lambda xi: compute_jacobian_determinant(
                compute_jacobian_matrix(
                    self.shape_functions(xi)[1].reshape(-1, 1),
                    self.node_coordinates_array[self.connectivity_array[self.element_index] - 1].reshape(-1, 1)
                )
            ),
            dim=1
        )

        dof_map = [0, 1, 2, 6, 7, 8]  
        self.Ke = expand_dof_mapping(Ke_reduced, full_size=12, dof_map=dof_map) 

    def validate_matrices(self):
        """
        Validates that Ke and Fe have the correct dimensions.
        """
        assert self.Ke.shape == (12, 12), f"Ke shape mismatch: Expected (12,12), got {self.Ke.shape}"
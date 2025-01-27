# pre_processing/element_library/element_1D_base.py

import numpy as np

# Import available subclasses dynamically
from pre_processing.element_library.euler_bernoulli.euler_bernoulli_beam_element import EulerBernoulliBeamElement
# Add other element subclasses here as needed, e.g.,
# from pre_processing.element_library.timoshenko.timoshenko_beam_element import TimoshenkoBeamElement

# Dictionary mapping element type names to their respective subclasses
ELEMENT_TYPE_MAP = {
    "EulerBernoulliBeamElement": EulerBernoulliBeamElement,
    # Add other element types here, e.g., "TimoshenkoBeamElement": TimoshenkoBeamElement,
}


class Element1DBase:
    """
    Base class for 1D finite elements. Provides common functionalities
    and data access methods for subclasses.

    Attributes:
        geometry_array (np.ndarray): Geometry properties array of shape (1, 20).
        material_array (np.ndarray): Material properties array of shape (1, 4).
        mesh_dictionary (dict): Dictionary containing mesh data.
        loads_array (np.ndarray): Loads array of shape (N, 9), where N is the number of nodes.
        dof_per_node (int): Degrees of freedom per node (default: 6).
    """

    def __init__(self, geometry_array: np.ndarray, material_array: np.ndarray,
                 mesh_dictionary: dict, loads_array: np.ndarray, dof_per_node: int = 6):
        """
        Initializes the base 1D finite element with parsed data.

        Args:
            geometry_array (np.ndarray): Geometry properties array of shape (1, 20).
            material_array (np.ndarray): Material properties array of shape (1, 4).
            mesh_dictionary (dict): Dictionary containing mesh data.
            loads_array (np.ndarray): Loads array of shape (N, 9), where N is the number of nodes.
            dof_per_node (int, optional): Degrees of freedom per node (default: 6).
        """
        self.geometry_array = geometry_array
        self.material_array = material_array
        self.mesh_dictionary = mesh_dictionary
        self.loads_array = loads_array
        self.dof_per_node = dof_per_node

        # Extract mesh information
        self._element_types = self.mesh_dictionary.get('element_types', np.array([], dtype=str))
        self._element_ids = self.mesh_dictionary.get('element_ids', np.array([], dtype=int))
        self._node_ids = self.mesh_dictionary.get('node_ids', np.array([], dtype=int))
        self._node_coordinates = self.mesh_dictionary.get('node_coordinates', np.array([], dtype=float))
        self._connectivity = self.mesh_dictionary.get('connectivity', np.array([], dtype=int))
        self._element_lengths = self.mesh_dictionary.get('element_lengths', np.array([], dtype=float))

        # Validate data shapes and consistency
        self._validate_mesh_data()

        # Precompute a mapping from element_id to index for O(1) access
        self._element_id_to_index = {eid: idx for idx, eid in enumerate(self._element_ids)}

    @property
    def element_types(self) -> np.ndarray:
        """Returns the array of element types."""
        return self._element_types

    @property
    def element_ids(self) -> np.ndarray:
        """Returns the array of element IDs."""
        return self._element_ids

    @property
    def node_ids(self) -> np.ndarray:
        """Returns the array of node IDs."""
        return self._node_ids

    @property
    def node_coordinates(self) -> np.ndarray:
        """Returns the array of node coordinates."""
        return self._node_coordinates

    @property
    def connectivity(self) -> np.ndarray:
        """Returns the connectivity array."""
        return self._connectivity

    @property
    def element_lengths(self) -> np.ndarray:
        """Returns the array of element lengths."""
        return self._element_lengths

    def _validate_mesh_data(self):
        """
        Validates the consistency and integrity of mesh data.
        """
        assert self._element_types.ndim == 1, "element_types should be a 1D array."
        assert self._element_ids.ndim == 1, "element_ids should be a 1D array."
        assert self._node_ids.ndim == 1, "node_ids should be a 1D array."
        assert self._node_coordinates.ndim == 2 and self._node_coordinates.shape[1] == 3, \
            "node_coordinates should be a 2D array with shape (N, 3)."
        assert self._connectivity.ndim == 2 and self._connectivity.shape[1] == 2, \
            "connectivity should be a 2D array with shape (M, 2)."
        assert self._element_lengths.ndim == 1, "element_lengths should be a 1D array."

        # Ensure that the number of elements matches across relevant arrays
        num_elements = len(self._element_ids)
        assert len(self._element_types) == num_elements, \
            "Mismatch between number of element_types and element_ids."
        assert len(self._connectivity) == num_elements, \
            "Mismatch between number of connectivity entries and element_ids."
        assert len(self._element_lengths) == num_elements, \
            "Mismatch between number of element_lengths and element_ids."

    def get_element_index(self, element_id: int) -> int:
        """
        Retrieves the zero-based index of the element based on its element_id.

        Args:
            element_id (int): The ID of the element.

        Returns:
            int: Zero-based index corresponding to the element's position in the mesh arrays.

        Raises:
            ValueError: If the element_id is not found in element_ids.
        """
        try:
            return self._element_id_to_index[element_id]
        except KeyError:
            raise ValueError(f"Element ID {element_id} not found in element_ids.")

    def instantiate_element(self, element_id: int):
        """
        Instantiates and returns an element object based on its type.

        Args:
            element_id (int): The ID of the element to instantiate.

        Returns:
            Element1DBase: An instance of a subclass corresponding to the element's type.

        Raises:
            ValueError: If the element_type is not recognized.
        """
        element_index = self.get_element_index(element_id)
        element_type = self.element_types[element_index]

        if element_type not in ELEMENT_TYPE_MAP:
            raise ValueError(f"Element type '{element_type}' is not supported.")

        # Instantiate the appropriate subclass
        element_class = ELEMENT_TYPE_MAP[element_type]
        element_instance = element_class(
            element_id=element_id,
            material_array=self.material_array,
            geometry_array=self.geometry_array,
            mesh_data=self.mesh_dictionary,
            node_positions=self.node_coordinates,
            loads_array=self.loads_array
        )
        return element_instance

    def assemble_global_dof_indices(self, element_id: int) -> list:
        """
        Assembles the global degree of freedom (DOF) indices for the element.

        Args:
            element_id (int): The ID of the element.

        Returns:
            list: List of global DOF indices for the element.
                  Example for 2-node, 6 DOF/node: [0,1,2,3,4,5,6,7,8,9,10,11]
        """
        node_ids = self.connectivity[self.get_element_index(element_id)]  # Shape: (2,)
        global_dof_indices = []
        for node_id in node_ids:
            # Assuming node numbering starts at 1 and global DOFs start at 0
            start_dof = (node_id - 1) * self.dof_per_node
            dof_indices = list(range(start_dof, start_dof + self.dof_per_node))
            global_dof_indices.extend(dof_indices)
        return global_dof_indices  # Length: num_nodes_per_element * dof_per_node


    def validate_matrices(self, element):
        """
        Validates that Ke and Fe have been computed and have correct dimensions.

        Args:
            element (object): The element instance to validate. Expected to have 'Ke' and 'Fe' attributes.

        Raises:
            AssertionError: If Ke or Fe do not have expected dimensions.
        """
        expected_Ke_shape = (self.dof_per_node * 2, self.dof_per_node * 2)  # e.g., (12, 12)
        expected_Fe_shape = (self.dof_per_node * 2,)  # e.g., (12,)
        assert hasattr(element, 'Ke'), "Stiffness matrix Ke is not computed."
        assert hasattr(element, 'Fe'), "Force vector Fe is not computed."
        assert element.Ke.shape == expected_Ke_shape, f"Ke shape mismatch: Expected {expected_Ke_shape}, got {element.Ke.shape}"
        assert element.Fe.shape == expected_Fe_shape, f"Fe shape mismatch: Expected {expected_Fe_shape}, got {element.Fe.shape}"



# ===========================
# Standalone Direct Test (Vectorized)
# ===========================
if __name__ == "__main__":
    """
    Fully vectorized standalone test that selects the appropriate element subclass
    and computes Ke and Fe for all elements.

    The test treats the element subclass as a black box. The subclass 
    determines the required inputs and performs internal calculations.
    """

    # Step 1: Parse data once for all elements
    parser = ParserBase()
    mesh_data = parser.mesh_data
    element_types = mesh_data['element_types']  # Array of element types

    num_elements = len(mesh_data['element_ids'])  # Total number of elements

    # Step 2: Vectorized selection of element subclasses
    def select_element_class(element_type):
        """ Returns the corresponding subclass for a given element type. """
        if element_type in ELEMENT_TYPE_MAP:
            return ELEMENT_TYPE_MAP[element_type]
        raise ValueError(f"Unknown element type: {element_type}")

    ElementClasses = np.vectorize(select_element_class)(element_types)  # Vectorized element class selection

    # Step 3: Vectorized initialization of all elements (Treating subclass as a black box)
    elements = np.array([
        ElementClass(
            element_id=i,
            material=parser.material_array,  # Pass the entire material array (black box input)
            section_props=parser.geometry_array,  # Pass the entire geometry array (black box input)
            mesh_data=mesh_data,
            node_positions=mesh_data['node_coordinates'],
            loads_array=parser.loads_array
        ) for i, ElementClass in enumerate(ElementClasses)
    ])

    # Step 4: Vectorized computation of Ke and Fe for all elements (Delegated to subclass)
    np.vectorize(lambda e: e.element_stiffness_matrix())(elements)
    np.vectorize(lambda e: e.element_force_vector())(elements)
    np.vectorize(lambda e: e.validate_matrices())(elements)

    print("\n===========================")
    print("1D Finite Element Direct Test (Vectorized)")
    print("===========================\n")

    # Step 5: Display results for first few elements
    for element in elements[:5]:  # Show first 5 elements
        print(f"\n--- Element {element.element_id + 1} ---")
        print("Element Type:", element_types[element.element_id])
        print("Connectivity:", element.get_element_connectivity())
        print("Node Coordinates:", element.get_node_coordinates())
        print("Element Length:", element.get_element_length())
        print("Element Loads:", element.get_element_loads())
        print("Stiffness Matrix (12×12):\n", element.Ke)
        print("Force Vector (1×12):\n", element.Fe)
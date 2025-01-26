# pre_processing/element_library/element_1D_base.py

import numpy as np

class Element1DBase:
    """
    Base class for 1D finite elements.

    Attributes:
        element_id (int): Unique identifier for the element.
        material (dict): Material properties dictionary.
        section_props (dict): Section properties dictionary.
        mesh_data (dict): Mesh data dictionary containing connectivity and element lengths.
        node_positions (ndarray): Array of node positions. Shape: (num_nodes,)
        loads_array (ndarray): Global loads array. Shape: (num_nodes, 6)
        dof_per_node (int): Degrees of freedom per node.
        Ke (ndarray): Element stiffness matrix.
        Fe (ndarray): Element force vector.
    """

    def __init__(self, element_id, material, section_props, mesh_data, node_positions, loads_array, dof_per_node=3):
        """
        Initializes the base 1D finite element.

        Args:
            element_id (int): Unique identifier for the element.
            material (dict): Material properties dictionary.
            section_props (dict): Section properties dictionary.
            mesh_data (dict): Mesh data dictionary containing connectivity and element lengths.
            node_positions (ndarray): Array of node positions. Shape: (num_nodes,)
            loads_array (ndarray): Global loads array. Shape: (num_nodes, 6)
            dof_per_node (int, optional): Degrees of freedom per node. Defaults to 3.
        """
        self.element_id = element_id
        self.material = material
        self.section_props = section_props
        self.mesh_data = mesh_data
        self.node_coordinates = node_positions
        self.loads_array = loads_array
        self.dof_per_node = dof_per_node
        self.Ke = None  # Element stiffness matrix (to be computed)
        self.Fe = None  # Element force vector (to be computed)

    def get_element_connectivity(self):
        """
        Retrieves the node IDs connected to this element.

        Returns:
            tuple: (start_node_id, end_node_id)
        """
        return self.mesh_data['connectivity'][self.element_id]

    def get_node_coordinates(self):
        """
        Retrieves the coordinates of the nodes connected to this element.

        Returns:
            ndarray: Array of node positions. Shape: (2,)
        """
        node_ids = self.get_element_connectivity()
        # Adjusting for zero-based indexing: node_id 1 corresponds to index 0
        return np.array([self.node_positions[node_id - 1] for node_id in node_ids])

    def get_element_length(self):
        """
        Retrieves the length of this element.

        Returns:
            float: Length of the element.
        """
        return float(self.mesh_data['element_lengths'][self.element_id])  # Ensures scalar

    def get_element_loads(self):
        """
        Retrieves the loads applied to this element's nodes.

        Returns:
            ndarray: Loads for start and end nodes. Shape: (2, 6)
        """
        node_ids = self.get_element_connectivity()
        # Adjusting for zero-based indexing
        return self.loads_array[[node_ids[0] - 1, node_ids[1] - 1], :]  # Shape: (2,6)
    
# Standalone execution for testing
if __name__ == "__main__":
    """
    Standalone test for the Element1DBase class.
    This will execute only if the script is run directly.
    """

    # Mock data for a simple 1D finite element
    element_id = 0
    material_array =   # Young's modulus (Pa), Density (kg/m^3)
    geometry_array =  # Cross-sectional area (m^2)

    # Mesh data
    node_ids_array = np.array([1, 2])
    node_coordinates_array = np.array([0.0, 1.0])  # 1D element with nodes at x = 0 and x = 1
    connectivity_array = {0: (1, 2)}  # Element 0 connects Node 1 and Node 2
    element_lengths_array = {0: 1.0}  # Element length = 1m

    mesh_dictionary = {
        'node_ids': node_ids_array,
        'node_coordinates': node_coordinates_array,
        'connectivity': connectivity_array,
        'element_lengths': element_lengths_array
    }

    # Loads applied at nodes (6 DOF per node: Fx, Fy, Fz, Mx, My, Mz)
    loads_array = np.zeros((2, 6))  # No loads applied

    # Degrees of freedom per node
    dof_per_node = 3  # Translational DOFs in 1D case

    # Initialize the base element
    element = Element1DBase(
        element_id=element_id,
        material=material,
        section_props=section_props,
        mesh_data=mesh_dictionary,
        node_positions=node_coordinates_array,
        loads_array=loads_array,
        dof_per_node=dof_per_node
    )

    # Print test results
    print("=== Element1DBase Standalone Test ===")
    print("Element ID:", element.element_id)
    print("Connectivity:", element.get_element_connectivity())
    print("Node Coordinates:", element.get_node_coordinates())
    print("Element Length:", element.get_element_length())
    print("Element Loads:", element.get_element_loads())
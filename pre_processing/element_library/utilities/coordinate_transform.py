# pre_processing\element_library\utilities\coordinate_transform.py

import numpy as np

def natural_to_physical(xi_array: np.ndarray, element_index: int, mesh_data: dict, element_lengths_array: np.ndarray) -> np.ndarray:
    """
    Converts natural coordinates to physical coordinates for 1D, 2D, or 3D elements.

    Parameters:
    - xi_array (np.ndarray): Natural coordinate(s) as a NumPy array.
        - 1D element: xi_array.shape = (N, 1) where N is the number of evaluation points.
        - 2D element: xi_array.shape = (N, 2) for bilinear/quadratic elements.
        - 3D element: xi_array.shape = (N, 3) for hexahedral or tetrahedral elements.
    - element_index (int): Index of the element.
    - mesh_data (dict): Dictionary containing:
        - 'elements': Indices of element nodes.
        - 'node_coordinates': Global coordinates of mesh nodes.
    - element_lengths_array (np.ndarray): Array of element lengths (for 1D elements).

    Returns:
    - np.ndarray: Physical coordinates in global space.
        - 1D: Returns (N, 1) array of x-coordinates.
        - 2D: Returns (N, 2) array of (x, y) coordinates.
        - 3D: Returns (N, 3) array of (x, y, z) coordinates.
    """

    # Retrieve element nodes in global coordinates
    element_nodes = mesh_data['elements'][element_index]  # Indices of element nodes
    node_coords = mesh_data['node_coordinates'][element_nodes]  # Shape: (n_nodes, dim)

    # Number of nodes and problem dimension
    n_nodes, dim = node_coords.shape

    # Vectorized computation for 1D case
    if dim == 1:
        element_length = element_lengths_array[element_index]
        x_phys = 0.5 * (xi_array + 1) * element_length  # Broadcast operation
        return x_phys.reshape(-1, 1)  # Keep shape consistent across dimensions

    # 2D and 3D elements: Compute shape functions and use matrix multiplication
    #elif dim == 2:
    #    N = shape_functions_2D(xi_array)  # Shape: (N, n_nodes)
    #    return np.dot(N, node_coords)  # Shape: (N, 2)

    #elif dim == 3:
    #    N = shape_functions_3D(xi_array)  # Shape: (N, n_nodes)
    #    return np.dot(N, node_coords)  # Shape: (N, 3)

    else:
        raise ValueError("Unsupported element dimensionality: dim must be 1")
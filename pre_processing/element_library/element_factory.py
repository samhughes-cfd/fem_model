# pre_processing\element_library\element_factory.py

import importlib
import numpy as np

# Registry mapping element names to module paths
ELEMENT_CLASS_MAP = {
    "EulerBernoulliBeamElement3DOF": "pre_processing.element_library.euler_bernoulli.euler_bernoulli_3DOF"
}

def create_elements_batch(mesh_dictionary, params_list):
    """
    Instantiates multiple finite elements in a batch using a fully vectorized approach.

    Args:
        mesh_dictionary (dict): Dictionary containing mesh data, including:
            - "element_types" (np.ndarray[str]): Array specifying element types for each element.
            - "connectivity" (np.ndarray[int]): Node connectivity information for each element.
            - "element_ids" (np.ndarray[int]): Unique identifiers for each element.
            - "element_lengths" (np.ndarray[float]): Element length data.
        params_list (np.ndarray[object]): NumPy array of dictionaries containing additional parameters for each element.

    Returns:
        np.ndarray: NumPy array of instantiated element objects.
    
    Raises:
        ValueError: If an unrecognized element type is found.
        ImportError: If a module for an element type cannot be imported.
        AttributeError: If the class for an element type is not found within its module.
    """
    element_types_array = mesh_dictionary["element_types"]
    element_ids_array = mesh_dictionary["element_ids"]

    # Identify unique element types (avoid redundant imports)
    unique_types = np.unique(element_types_array)
    modules = {}

    for etype in unique_types:
        if etype not in ELEMENT_CLASS_MAP:
            raise ValueError(f"Unknown element type: {etype}. Check ELEMENT_CLASS_MAP for valid types.")

        module_name = ELEMENT_CLASS_MAP[etype]
        try:
            modules[etype] = importlib.import_module(module_name)  # Import module dynamically
        except ImportError as e:
            raise ImportError(f"Module '{module_name}' could not be imported: {e}")

    # **Vectorized class selection**
    try:
        class_references = np.array(
            [getattr(modules[etype], etype) for etype in element_types_array], dtype=object
        )
    except AttributeError as e:
        raise AttributeError(f"Failed to find class for element type: {etype}. Ensure the class name matches.")

    # **Vectorized instantiation using NumPy mapping**
    elements = np.fromiter(
    (cls(element_id=elem_id, **params)  # mesh_dictionary is already inside params_list
     for cls, elem_id, params in zip(class_references, element_ids_array, params_list)),
    dtype=object,
    count=len(element_ids_array)
    )


    return elements
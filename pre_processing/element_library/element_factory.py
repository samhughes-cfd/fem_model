# pre_processing\element_library\element_factory.py

import importlib
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

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
            - "element_ids" (np.ndarray[int]): Unique identifiers for each element.
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
            logger.error(f"Unknown element type: {etype}. Check ELEMENT_CLASS_MAP for valid types.")
            raise ValueError(f"Unknown element type: {etype}. Check ELEMENT_CLASS_MAP for valid types.")

        module_name = ELEMENT_CLASS_MAP[etype]
        try:
            modules[etype] = importlib.import_module(module_name)  # Import module dynamically
            logger.info(f"Successfully imported module: {module_name}")
        except ImportError as e:
            logger.error(f"Module '{module_name}' could not be imported: {e}")
            raise ImportError(f"Module '{module_name}' could not be imported: {e}")

    # **Vectorized class selection**
    class_references = []
    for etype in element_types_array:
        try:
            class_references.append(getattr(modules[etype], etype))
        except AttributeError as e:
            logger.error(f"Failed to find class '{etype}' in module '{modules[etype]}'. Ensure the class name matches.")
            raise AttributeError(f"Failed to find class '{etype}' in module '{modules[etype]}'. Ensure the class name matches.") from e

    class_references = np.array(class_references, dtype=object)

    # **Prevent Infinite Loops: Track Created Elements**
    seen_elements = set()
    elements = []

    for cls, elem_id, params in zip(class_references, element_ids_array, params_list):
        if elem_id in seen_elements:
            logger.warning(f"Skipping duplicate instantiation of element {elem_id}.")
            continue  # Skip re-creating elements

        seen_elements.add(elem_id)
        try:
            elements.append(cls(element_id=elem_id, **params))
        except Exception as e:
            logger.error(f"Error instantiating element {elem_id} of type {cls.__name__}: {e}")
            elements.append(None)  # Preserve array shape even if an element fails

    return np.array(elements, dtype=object)  # Convert list to NumPy array
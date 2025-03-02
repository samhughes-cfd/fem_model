# pre_processing\element_library\element_factory.py

import importlib
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Registry mapping element names to module paths
ELEMENT_CLASS_MAP = {
    "EulerBernoulliBeamElement3DOF": "pre_processing.element_library.euler_bernoulli.euler_bernoulli_3DOF",
    "EulerBernoulliBeamElement6DOF": "pre_processing.element_library.euler_bernoulli.euler_bernoulli_6DOF"  
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

    # Check for unknown element types before proceeding
    missing_types = [etype for etype in unique_types if etype not in ELEMENT_CLASS_MAP]
    if missing_types:
        logger.error(f"❌ Unrecognized element types found: {missing_types}")
        raise ValueError(f"❌ Unrecognized element types: {missing_types}. Update ELEMENT_CLASS_MAP.")

    for etype in unique_types:
        module_name = ELEMENT_CLASS_MAP[etype]
        try:
            modules[etype] = importlib.import_module(module_name)  # Import module dynamically
            logger.info(f"✅ Successfully imported module: {module_name}")
        except ImportError as e:
            logger.exception(f"❌ Failed to import '{module_name}'. Verify module existence and PYTHONPATH.")
            raise

    # **Vectorized class selection**
    try:
        class_references = np.array([getattr(modules[etype], etype) for etype in element_types_array], dtype=object)
    except AttributeError as e:
        logger.error(f"❌ Failed to find element class: {e}")
        raise AttributeError(f"❌ Failed to find element class. Ensure class names match module names.") from e

    # **Vectorized instantiation using np.vectorize**
    def instantiate_element(cls, elem_id, params):
        try:
            return cls(element_id=elem_id, **params)
        except Exception as e:
            logger.error(f"❌ Error instantiating element {elem_id} of type {cls.__name__}: {e}")
            return None  # Preserve array shape

    vectorized_instantiation = np.vectorize(instantiate_element, otypes=[object])
    elements = vectorized_instantiation(class_references, element_ids_array, params_list)

    return elements
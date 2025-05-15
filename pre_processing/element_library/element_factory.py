# pre_processing/element_library/element_factory.py

import importlib
import numpy as np
import logging
import os
from pathvalidate import sanitize_filename

# Configure logging
logger = logging.getLogger(__name__)

# Registry mapping element names to module paths
ELEMENT_CLASS_MAP = {
    "EulerBernoulliBeamElement6DOF": "pre_processing.element_library.euler_bernoulli.euler_bernoulli_6DOF"  
}

def create_elements_batch(mesh_dictionary, params_list):
    """
    Robust batch element instantiation with logging validation.
    
    Ensures:
    - All elements have valid logger operators
    - Required logging directories exist
    - Element IDs are safely sanitized
    """
    element_types_array = mesh_dictionary["element_types"]
    element_ids_array = mesh_dictionary["element_ids"]

    # Validate input parameters
    if not isinstance(params_list, np.ndarray) or params_list.dtype != object:
        raise ValueError("params_list must be object-type numpy array")

    # Check directory existence in parameters
    if any("job_results_dir" not in p or not p["job_results_dir"] for p in params_list):
        logger.error("Missing job_results_dir in element parameters")
        raise ValueError("All elements require job_results_dir")

    # Identify and validate element types
    unique_types = np.unique(element_types_array)
    modules = _load_element_modules(unique_types)

    # Create elements with strict validation
    elements = _instantiate_elements(
        element_types_array,
        element_ids_array,
        params_list,
        modules
    )

    # Post-instantiation validation
    _validate_element_logging(elements, params_list[0]["job_results_dir"])
    
    return elements

def _load_element_modules(unique_types):
    """Load required element modules with validation"""
    modules = {}
    for etype in unique_types:
        if etype not in ELEMENT_CLASS_MAP:
            raise ValueError(f"Unregistered element type: {etype}")
            
        module_name = ELEMENT_CLASS_MAP[etype]
        try:
            modules[etype] = importlib.import_module(module_name)
            logger.info(f"Successfully imported {module_name}")
        except ImportError as e:
            logger.error(f"Module import failed: {module_name}")
            raise RuntimeError(f"Could not load {etype} module") from e
    return modules

def _instantiate_elements(element_types, element_ids, params_list, modules):
    """Safe element instantiation with error containment"""
    try:
        # Get class references with validation
        class_refs = np.array([
            _get_class_reference(modules[etype], etype)
            for etype in element_types
        ], dtype=object)

        # Vectorized instantiation with sanitized IDs
        elements = np.vectorize(
            _safe_instantiate,
            otypes=[object]
        )(class_refs, element_ids, params_list)

        # Check for instantiation failures
        if np.any(elements == None):
            null_indices = np.where(elements == None)[0]
            raise RuntimeError(f"Null elements at indices: {null_indices}")
            
        return elements
    except Exception as e:
        logger.error("Element batch creation failed")
        raise

def _get_class_reference(module, class_name):
    """Validate class existence in module"""
    cls = getattr(module, class_name, None)
    if not cls:
        raise AttributeError(f"Class {class_name} not found in {module.__name__}")
    return cls

def _safe_instantiate(cls, elem_id, params):
    """Element instantiation with logging validation"""
    try:
        # Sanitize element ID for filesystem safety
        sanitized_id = sanitize_filename(str(elem_id))
        params["element_id"] = sanitized_id
        
        # Create element and validate logger
        element = cls(**params)
        
        if not hasattr(element, "logger_operator") or not element.logger_operator:
            raise ValueError(f"Element {elem_id} missing logger operator")
            
        return element
    except Exception as e:
        logger.error(f"Failed to create element {elem_id}: {str(e)}")
        raise  # Re-raise to stop execution

def _validate_element_logging(elements, job_results_dir):
    """Post-creation validation of logging infrastructure"""
    required_dirs = [
        os.path.join(job_results_dir, "element_stiffness_matrices"),
        os.path.join(job_results_dir, "element_force_vectors")
    ]
    
    # Verify directories exist
    for d in required_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing logging directory: {d}")
        if not os.access(d, os.W_OK):
            raise PermissionError(f"Cannot write to directory: {d}")

    # Verify element logging configuration
    for idx, element in enumerate(elements):
        if not element.logger_operator:
            raise ValueError(f"Element {idx} missing logger operator")
        if element.logger_operator.job_results_dir != job_results_dir:
            raise ValueError(f"Element {idx} directory mismatch")
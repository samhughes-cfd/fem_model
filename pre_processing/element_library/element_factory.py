# pre_processing/element_library/element_factory.py

import importlib
import numpy as np
import logging
import os
from pathvalidate import sanitize_filename
from typing import List, Dict, Any, TYPE_CHECKING
from pre_processing.element_library.base_logger_operator import BaseLoggerOperator, LoggingConfigurationError

# TYPE_CHECKING import for static analysis only
if TYPE_CHECKING:
    from pre_processing.element_library.element_1D_base import Element1DBase

# Configure logging
logger = logging.getLogger(__name__)

# Registry mapping element names to module paths
ELEMENT_CLASS_MAP = {
    "EulerBernoulliBeamElement3D": "pre_processing.element_library.euler_bernoulli.euler_bernoulli_3D"  
}

def create_elements_batch(mesh_dictionary: Dict[str, Any], params_list: List[Dict[str, Any]]) -> List['Element1DBase']:
    """
    Precision-aware element factory with comprehensive validation pipeline
    
    Features:
    - Strict type checking for numerical inputs
    - Filesystem safety checks
    - Logging infrastructure validation
    - Precision configuration enforcement
    """
    # Runtime import to break circular dependency
    from pre_processing.element_library.element_1D_base import Element1DBase

    # === NEW: PHASE 0 - PRE-VALIDATION TYPE HARDENING ===
    element_ids_array = _sanitize_element_ids(mesh_dictionary["element_ids"])
    
    # Enforce ID consistency between mesh and parameters
    if len(element_ids_array) != len(params_list):
        raise ValueError(f"Element ID count mismatch: {len(element_ids_array)} IDs vs {len(params_list)} parameter sets")

    # Convert all element IDs to native int type first
    for param, elem_id in zip(params_list, element_ids_array):
        param["element_id"] = int(elem_id)  # Force native Python int

    # === ORIGINAL VALIDATION PIPELINE WITH ENHANCEMENTS ===
    # Phase 1: Input validation
    _validate_mesh_dictionary(mesh_dictionary)
    _validate_params_structure(params_list)
    
    element_types_array = np.asarray(mesh_dictionary["element_types"])
    
    # Phase 2: Environment validation
    job_dir = _validate_job_environment(params_list)
    
    # Phase 3: Module loading
    modules = _load_element_modules(element_types_array)
    
    # Phase 4: Element instantiation
    elements = _instantiate_elements(
        element_types_array,
        element_ids_array,
        params_list,
        modules
    )
    
    # Phase 5: Post-instantiation validation
    _validate_logging_infrastructure(elements, job_dir)
    
    return elements

def _validate_mesh_dictionary(mesh_dict: Dict[str, Any]) -> None:
    """Validate mesh structure and content types"""
    required_keys = {"element_types", "element_ids", "connectivity", "node_coordinates"}
    missing_keys = required_keys - mesh_dict.keys()
    if missing_keys:
        raise KeyError(f"Missing required mesh keys: {missing_keys}")
    
    if len(mesh_dict["element_ids"]) != len(mesh_dict["element_types"]):
        raise ValueError("Element IDs and types array size mismatch")

def _validate_params_structure(params_list: List[Dict[str, Any]]) -> None:
    """Validate parameters structure and required fields"""
    required_params = {
        "geometry_array", "material_array", "mesh_dictionary",
        "point_load_array", "distributed_load_array",
        "job_results_dir", "element_id"
    }
    for params in params_list:
        missing = required_params - params.keys()
        if missing:
            raise KeyError(f"Missing required parameters: {missing}")
            
        # Enhanced type checking
        type_checks = [
            ("geometry_array", np.ndarray),
            ("material_array", np.ndarray),
            ("mesh_dictionary", dict),
            ("point_load_array", np.ndarray),
            ("distributed_load_array", np.ndarray),
            ("element_id", (int, np.integer)),  # Now validated after conversion
            ("job_results_dir", str)
        ]
        
        for param_name, param_type in type_checks:
            if not isinstance(params[param_name], param_type):
                raise TypeError(
                    f"{param_name} must be {param_type}, got {type(params[param_name])}"
                )

# === ENHANCED SANITIZATION FUNCTION ===
def _sanitize_element_ids(raw_ids: List[Any]) -> np.ndarray:
    """Convert and validate element IDs with enhanced parsing"""
    converted = []
    for idx, raw_id in enumerate(raw_ids):
        try:
            # Handle string representations of numbers
            if isinstance(raw_id, str):
                # Strip whitespace and check numeric
                clean_id = raw_id.strip()
                if not clean_id.isdigit():
                    raise ValueError(f"Non-numeric ID at position {idx}: '{raw_id}'")
                converted.append(int(clean_id))
            else:
                # Convert to int regardless of original type
                converted.append(int(raw_id))
        except (ValueError, TypeError) as e:
            raise TypeError(f"Invalid element ID at index {idx}: {repr(raw_id)}") from e
    
    ids_array = np.array(converted, dtype=np.int64)
    if np.any(ids_array < 0):
        raise ValueError("Negative element IDs found")
    return ids_array

def _validate_job_environment(params_list: List[Dict[str, Any]]) -> str:
    """Validate shared job directory configuration"""
    directories = {p["job_results_dir"] for p in params_list}
    if len(directories) > 1:
        raise ValueError("All elements must share the same job_results_dir")
    
    job_dir = directories.pop()
    if not os.path.isdir(job_dir):
        raise FileNotFoundError(f"Job directory {job_dir} missing")
    if not os.access(job_dir, os.W_OK):
        raise PermissionError(f"Write access denied for {job_dir}")
    
    return job_dir

def _load_element_modules(element_types: np.ndarray) -> Dict[str, Any]:
    """Load element modules with dependency validation"""
    modules = {}
    for etype in np.unique(element_types):
        if etype not in ELEMENT_CLASS_MAP:
            raise ValueError(f"Unregistered element type: {etype}")
            
        try:
            module = importlib.import_module(ELEMENT_CLASS_MAP[etype])
            if not hasattr(module, etype):
                raise AttributeError(f"Module missing class {etype}")
            modules[etype] = module
        except ImportError as e:
            logger.critical("Module load failed for %s", etype)
            raise RuntimeError(f"Critical dependency error: {etype}") from e
    
    return modules

def _instantiate_elements(
    element_types: np.ndarray,
    element_ids: np.ndarray,
    params_list: List[Dict[str, Any]],
    modules: Dict[str, Any]
) -> List['Element1DBase']:
    """Safe element instantiation pipeline"""
    elements = []
    for idx, (etype, eid, params) in enumerate(zip(element_types, element_ids, params_list)):
        try:
            cls = getattr(modules[etype], etype)
            
            # === REDUNDANT TYPE ENFORCEMENT ===
            init_params = {
                "geometry_array": params["geometry_array"],
                "material_array": params["material_array"].astype(np.float64),
                "mesh_dictionary": params["mesh_dictionary"],
                "point_load_array": params["point_load_array"],
                "distributed_load_array": params["distributed_load_array"],
                "job_results_dir": str(params["job_results_dir"]),
                # Final type guarantee
                "element_id": int(params["element_id"])  
            }
            
            # Conditionally add optional parameters
            optional_params = {
                "dof_per_node": params.get("dof_per_node"),
                "quadrature_order": params.get("quadrature_order")
            }
            
            init_params.update({k: v for k, v in optional_params.items() if v is not None})

            # Runtime inheritance check
            from pre_processing.element_library.element_1D_base import Element1DBase
            if not issubclass(cls, Element1DBase):
                raise TypeError(f"{cls.__name__} must inherit from Element1DBase")

            element = cls(**init_params)
            
            elements.append(element)
            
        except Exception as e:
            logger.error("Element %d (ID %d) creation failed", idx, int(eid))
            logger.debug("Failure details:", exc_info=True)
            raise RuntimeError(f"Element {int(eid)} instantiation failed") from e
    
    return elements

def _validate_logging_infrastructure(elements: List['Element1DBase'], job_dir: str) -> None:
    """Final validation of logging setup"""
    # Validate directory structure
    required_subdirs = ["element_stiffness_matrices", "element_force_vectors"]
    for subdir in required_subdirs:
        dir_path = os.path.join(job_dir, subdir)
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Missing directory: {dir_path}")
        if not os.access(dir_path, os.W_OK):
            raise PermissionError(f"Write protected: {dir_path}")
    
    # Validate element logging capabilities
    for element in elements:
        logger = element.logger_operator
        try:
            # Test matrix logging
            test_matrix = np.zeros((2,2), dtype=np.float64)
            logger.log_matrix("stiffness", test_matrix, {"name": "Validation Matrix"})
            logger.flush_all()
            
            # Verify file creation
            log_path = logger._get_log_path("stiffness")
            if not os.path.isfile(log_path):
                raise FileNotFoundError(f"Log file missing: {log_path}")
            if os.path.getsize(log_path) == 0:
                raise IOError(f"Empty log file: {log_path}")
                
        except Exception as e:
            raise LoggingConfigurationError(
                f"Logging validation failed for element {element.element_id}"
            ) from e
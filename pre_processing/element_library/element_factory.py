# pre_processing/element_library/element_factory.py

import importlib
import numpy as np
import logging
import os
from pathvalidate import sanitize_filename
from typing import List, Dict, Any, TYPE_CHECKING
from pre_processing.element_library.base_logger_operator import BaseLoggerOperator, LoggingConfigurationError

if TYPE_CHECKING:
    from pre_processing.element_library.element_1D_base import Element1DBase

logger = logging.getLogger(__name__)

ELEMENT_CLASS_MAP = {
    "EulerBernoulliBeamElement3D": "pre_processing.element_library.euler_bernoulli.euler_bernoulli_3D"  
}

def create_elements_batch(mesh_dictionary: Dict[str, Any], params_list: List[Dict[str, Any]]) -> List['Element1DBase']:
    logger.info("🔧 Starting element batch creation for %d elements", len(params_list))
    from pre_processing.element_library.element_1D_base import Element1DBase

    element_ids_array = _sanitize_element_ids(mesh_dictionary["element_ids"])

    if len(element_ids_array) != len(params_list):
        logger.error("❌ Element ID count mismatch: %d IDs vs %d parameter sets", len(element_ids_array), len(params_list))
        raise ValueError("Element ID count mismatch")

    for param, elem_id in zip(params_list, element_ids_array):
        param["element_id"] = int(elem_id)

    _validate_mesh_dictionary(mesh_dictionary)
    _validate_params_structure(params_list)

    element_types_array = np.asarray(mesh_dictionary["element_types"])

    job_dir = _validate_job_environment(params_list)
    modules = _load_element_modules(element_types_array)

    elements = _instantiate_elements(
        element_types_array,
        element_ids_array,
        params_list,
        modules
    )

    _validate_logging_infrastructure(elements, job_dir)

    logger.info("✅ Successfully created and validated %d elements", len(elements))
    return elements

def _validate_mesh_dictionary(mesh_dict: Dict[str, Any]) -> None:
    required_keys = {"element_types", "element_ids", "connectivity", "node_coordinates"}
    missing_keys = required_keys - mesh_dict.keys()
    if missing_keys:
        logger.critical("❌ Missing required mesh keys: %s", missing_keys)
        raise KeyError(f"Missing required mesh keys: {missing_keys}")

    if len(mesh_dict["element_ids"]) != len(mesh_dict["element_types"]):
        logger.error("❌ Element IDs and types array size mismatch")
        raise ValueError("Element IDs and types array size mismatch")

def _validate_params_structure(params_list: List[Dict[str, Any]]) -> None:
    required_params = {
        "geometry_array", "material_array", "mesh_dictionary",
        "point_load_array", "distributed_load_array",
        "job_results_dir", "element_id"
    }
    for idx, params in enumerate(params_list):
        missing = required_params - params.keys()
        if missing:
            logger.error("❌ Element %d missing required parameters: %s", idx, missing)
            raise KeyError(f"Missing required parameters: {missing}")

        type_checks = [
            ("geometry_array", np.ndarray),
            ("material_array", np.ndarray),
            ("mesh_dictionary", dict),
            ("point_load_array", np.ndarray),
            ("distributed_load_array", np.ndarray),
            ("element_id", (int, np.integer)),
            ("job_results_dir", str)
        ]

        for param_name, param_type in type_checks:
            if not isinstance(params[param_name], param_type):
                logger.error("❌ Parameter '%s' in element %d is of incorrect type: expected %s, got %s", param_name, idx, param_type, type(params[param_name]))
                raise TypeError(f"{param_name} must be {param_type}, got {type(params[param_name])}")

def _sanitize_element_ids(raw_ids: List[Any]) -> np.ndarray:
    converted = []
    for idx, raw_id in enumerate(raw_ids):
        try:
            if isinstance(raw_id, str):
                clean_id = raw_id.strip()
                if not clean_id.isdigit():
                    logger.error("❌ Non-numeric element ID at index %d: '%s'", idx, raw_id)
                    raise ValueError(f"Non-numeric ID at position {idx}: '{raw_id}'")
                converted.append(int(clean_id))
            else:
                converted.append(int(raw_id))
        except (ValueError, TypeError) as e:
            logger.error("❌ Invalid element ID at index %d: %s", idx, raw_id, exc_info=True)
            raise

    ids_array = np.array(converted, dtype=np.int64)
    logger.debug("✅ Sanitized element IDs: %s", ids_array.tolist())
    if np.any(ids_array < 0):
        raise ValueError("Negative element IDs found")
    return ids_array

def _validate_job_environment(params_list: List[Dict[str, Any]]) -> str:
    directories = {p["job_results_dir"] for p in params_list}
    if len(directories) > 1:
        logger.critical("❌ Multiple job directories found: %s", directories)
        raise ValueError("All elements must share the same job_results_dir")

    job_dir = directories.pop()
    if not os.path.isdir(job_dir):
        logger.critical("❌ Job directory missing: %s", job_dir)
        raise FileNotFoundError(f"Job directory {job_dir} missing")
    if not os.access(job_dir, os.W_OK):
        logger.critical("❌ Write access denied to job directory: %s", job_dir)
        raise PermissionError(f"Write access denied for {job_dir}")

    logger.debug("📁 Job directory validated: %s", job_dir)
    return job_dir

def _load_element_modules(element_types: np.ndarray) -> Dict[str, Any]:
    modules = {}
    for etype in np.unique(element_types):
        if etype not in ELEMENT_CLASS_MAP:
            logger.error("❌ Unregistered element type: %s", etype)
            raise ValueError(f"Unregistered element type: {etype}")

        try:
            module = importlib.import_module(ELEMENT_CLASS_MAP[etype])
            if not hasattr(module, etype):
                raise AttributeError(f"Module missing class {etype}")
            modules[etype] = module
            logger.debug("✅ Loaded module for element type: %s", etype)
        except ImportError as e:
            logger.critical("❌ Import failed for element type '%s': %s", etype, e, exc_info=True)
            raise RuntimeError(f"Critical dependency error: {etype}") from e

    return modules

def _instantiate_elements(
    element_types: np.ndarray,
    element_ids: np.ndarray,
    params_list: List[Dict[str, Any]],
    modules: Dict[str, Any]
) -> List['Element1DBase']:
    elements = []
    for idx, (etype, eid, params) in enumerate(zip(element_types, element_ids, params_list)):
        try:
            cls = getattr(modules[etype], etype)
            init_params = {
                "geometry_array": params["geometry_array"],
                "material_array": params["material_array"].astype(np.float64),
                "mesh_dictionary": params["mesh_dictionary"],
                "point_load_array": params["point_load_array"],
                "distributed_load_array": params["distributed_load_array"],
                "job_results_dir": str(params["job_results_dir"]),
                "element_id": int(params["element_id"])
            }
            optional_params = {
                "dof_per_node": params.get("dof_per_node"),
                "quadrature_order": params.get("quadrature_order")
            }
            init_params.update({k: v for k, v in optional_params.items() if v is not None})

            from pre_processing.element_library.element_1D_base import Element1DBase
            if not issubclass(cls, Element1DBase):
                raise TypeError(f"{cls.__name__} must inherit from Element1DBase")

            element = cls(**init_params)
            logger.info("🧱 Instantiated element %d (type %s)", eid, etype)
            elements.append(element)

        except Exception as e:
            logger.error("❌ Failed to instantiate element ID %s (type %s): %s", eid, etype, e, exc_info=True)
            logger.debug("💥 Params snapshot: %s", {k: str(v)[:100] for k, v in params.items()})
            raise RuntimeError(f"Element {int(eid)} instantiation failed") from e

    return elements

def _validate_logging_infrastructure(elements: List['Element1DBase'], job_dir: str) -> None:
    required_subdirs = ["element_stiffness_matrices", "element_force_vectors"]
    for subdir in required_subdirs:
        dir_path = os.path.join(job_dir, subdir)
        if not os.path.isdir(dir_path):
            logger.error("❌ Missing directory for logging: %s", dir_path)
            raise FileNotFoundError(f"Missing directory: {dir_path}")
        if not os.access(dir_path, os.W_OK):
            logger.error("❌ Write protected log directory: %s", dir_path)
            raise PermissionError(f"Write protected: {dir_path}")

    for element in elements:
        logger_operator = element.logger_operator
        try:
            test_matrix = np.zeros((2,2), dtype=np.float64)
            logger_operator.log_matrix("stiffness", test_matrix, {"name": "Validation Matrix"})
            logger_operator.flush_all()
            log_path = logger_operator._get_log_path("stiffness")
            if not os.path.isfile(log_path) or os.path.getsize(log_path) == 0:
                raise IOError(f"Log file verification failed: {log_path}")
            logger.debug("✅ Logging test passed for element %s", element.element_id)
        except Exception as e:
            logger.error("❌ Logging infrastructure invalid for element %s: %s", element.element_id, e, exc_info=True)
            raise LoggingConfigurationError(
                f"Logging validation failed for element {element.element_id}"
            ) from e
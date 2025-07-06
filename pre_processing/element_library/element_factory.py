# pre_processing/element_library/element_factory.py

import importlib
import logging
import os
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid runtime circular import
    from pre_processing.element_library.element_1D_base import Element1DBase


class ElementFactory:
    """
    Batch-builds 1-D element objects and validates their logging infrastructure.

    Notes
    -----
    * Concrete element classes (e.g. ``EulerBernoulliBeamElement3D``) **must**
      inherit from
      :class:`pre_processing.element_library.element_1D_base.Element1DBase`.
    * Classes are discovered via :pyattr:`ELEMENT_CLASS_MAP`.
    * Per-element logs are written to ``<job_results_dir>/logs``.
    """

    ELEMENT_CLASS_MAP = {
        "EulerBernoulliBeamElement3D":
            "pre_processing.element_library.euler_bernoulli.euler_bernoulli_3D",
    }

    # ------------------------------------------------------------------ #
    def __init__(self, job_results_dir: str) -> None:
        """
        Parameters
        ----------
        job_results_dir
            Root directory where each element’s log files will be created.
        """
        self.job_results_dir = job_results_dir
        self.logger = self._setup_logger()

    # ------------------------------------------------------------------ #
    def create_elements_batch(
        self,
        *,
        element_ids: np.ndarray,
        element_dictionary: Dict[str, Any],
        grid_dictionary: Dict[str, Any],
        material_dictionary: Dict[str, Any],
        section_dictionary: Dict[str, Any],
        point_load_array: np.ndarray,
        distributed_load_array: np.ndarray,
    ) -> List["Element1DBase"]:
        """
        Instantiate one element object for every ID in *element_ids*.

        Parameters
        ----------
        element_ids
            NumPy ``int64`` array of element IDs. Order must match
            ``element_dictionary["types"]`` and ``["connectivity"]``.
        element_dictionary
            Parsed output from :class:`ElementParser`.
            Required keys: ``"ids"``, ``"types"``, ``"connectivity"``.
        grid_dictionary
            Parsed output from :class:`GridParser`.
        material_dictionary
            Parsed output from :class:`MaterialParser`.
        section_dictionary
            Parsed output from :class:`SectionParser`.
        point_load_array
            ``(N, 4)`` array or empty array of point-load data.
        distributed_load_array
            ``(N, 5)`` array or empty array of distributed-load data.

        Returns
        -------
        list[Element1DBase]
            One instance per ID in *element_ids*.
        """
        self.logger.info("🚩 Starting batch element creation.")

        element_ids_array = self._sanitize_element_ids(element_ids)
        self._validate_element_dictionary(element_dictionary)

        element_types_array = np.asarray(element_dictionary["types"])
        if len(element_types_array) != len(element_ids_array):
            self.logger.error(
                "Element ID / type count mismatch: IDs=%d, types=%d",
                len(element_ids_array), len(element_types_array),
            )
            raise ValueError("Mismatch between element IDs and types.")

        modules = self._load_element_modules(element_types_array)

        elements: List["Element1DBase"] = []
        for etype, eid in zip(element_types_array, element_ids_array):
            params = {
                "element_id":             int(eid),
                "element_dictionary":     element_dictionary,
                "grid_dictionary":        grid_dictionary,
                "material_dictionary":    material_dictionary,
                "section_dictionary":     section_dictionary,
                "point_load_array":       point_load_array,
                "distributed_load_array": distributed_load_array,
                "job_results_dir":        self.job_results_dir,
            }
            elem = self._instantiate_element(etype, eid, params, modules)
            elements.append(elem)
            self.logger.debug("✅ Element %s (%s) instantiated.", eid, etype)

        self._validate_logging_infrastructure(elements)
        self.logger.info("✅ Batch element creation complete.")
        return elements

    # ------------------------------------------------------------------ #
    # PRIVATE HELPERS
    # ------------------------------------------------------------------ #
    def _setup_logger(self) -> logging.Logger:
        """
        Create a dedicated file logger ``ElementFactory.log`` under
        ``<job_results_dir>/logs``. Duplicate handlers are purged to avoid
        repeated messages in multi-process runs.
        """
        log_dir = os.path.join(self.job_results_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        logger = logging.getLogger("ElementFactory")
        logger.setLevel(logging.DEBUG)
        if logger.handlers:
            logger.handlers.clear()

        fh = logging.FileHandler(
            os.path.join(log_dir, "ElementFactory.log"), mode="w", encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
        logger.addHandler(fh)
        logger.propagate = False

        logger.debug("🟢 ElementFactory logger initialised.")
        return logger

    # .................................................................. #
    def _sanitize_element_ids(self, raw_ids) -> np.ndarray:
        """
        Convert *raw_ids* to ``int64`` NumPy array and verify positivity.

        Raises
        ------
        ValueError
            If any ID is negative or non-numeric.
        """
        try:
            ids = np.asarray(raw_ids, dtype=np.int64)
            if np.any(ids < 0):
                raise ValueError("Negative element IDs detected.")
            self.logger.debug("Element IDs sanitised: %s", ids.tolist())
            return ids
        except Exception as exc:
            self.logger.error("Invalid element IDs.", exc_info=True)
            raise ValueError("Invalid element IDs") from exc

    # .................................................................. #
    def _validate_element_dictionary(self, element_dict: Dict[str, Any]) -> None:
        """
        Ensure *element_dict* has the mandatory keys ``ids``, ``types``,
        ``connectivity``.
        """
        required = {"types", "ids", "connectivity"}
        missing = required - element_dict.keys()
        if missing:
            self.logger.error("Missing keys in element dictionary: %s", missing)
            raise KeyError(f"Missing keys: {missing}")

    # .................................................................. #
    def _load_element_modules(self, element_types: np.ndarray) -> Dict[str, Any]:
        """
        Dynamically import every unique element module referenced in
        *element_types*.

        Returns
        -------
        dict
            Mapping ``element_type → imported module``.
        """
        modules: Dict[str, Any] = {}
        for etype in np.unique(element_types):
            if etype not in self.ELEMENT_CLASS_MAP:
                self.logger.error("Unregistered element type: %s", etype)
                raise ValueError(f"Unregistered element type: {etype}")

            module_path = self.ELEMENT_CLASS_MAP[etype]
            try:
                module = importlib.import_module(module_path)
                if not hasattr(module, etype):
                    raise AttributeError(
                        f"Module {module_path} missing class {etype}"
                    )
                modules[etype] = module
                self.logger.debug("Module for %s loaded.", etype)
            except Exception as exc:
                self.logger.error("Cannot load module %s", module_path, exc_info=True)
                raise ImportError(f"Cannot load module {module_path}") from exc
        return modules

    # .................................................................. #
    def _instantiate_element(
        self,
        etype: str,
        eid: int,
        params: Dict[str, Any],
        modules: Dict[str, Any],
    ) -> "Element1DBase":
        """
        Instantiate a single element given its type *etype* and constructor
        *params* pulled from the various dictionaries.
        """
        from pre_processing.element_library.element_1D_base import Element1DBase

        cls = getattr(modules[etype], etype)
        if not issubclass(cls, Element1DBase):
            raise TypeError(f"Class {etype} must inherit from Element1DBase.")

        return cls(**params)

    # .................................................................. #
    def _validate_logging_infrastructure(self, elements: List["Element1DBase"]) -> None:
        """
        Smoke-test each element’s `logger_operator` by writing a dummy matrix
        and confirming a non-empty file on disk.

        Raises
        ------
        RuntimeError
            If any element fails the logging check.
        """
        import numpy as np  # local to keep top-level deps minimal

        for elem in elements:
            try:
                log_op = elem.logger_operator
                log_op.log_matrix("stiffness", np.zeros((2, 2)), {"test": True})
                log_op.flush_all()
                path = log_op._get_log_path("stiffness")
                if not os.path.isfile(path) or os.path.getsize(path) == 0:
                    raise IOError(f"Logging failed for element {elem.element_id}")
                self.logger.debug("Logging verified for element %s.", elem.element_id)
            except Exception as exc:
                self.logger.error(
                    "Logging infrastructure invalid for element %s",
                    elem.element_id, exc_info=True,
                )
                raise RuntimeError(
                    f"Logging validation failed for element {elem.element_id}"
                ) from exc
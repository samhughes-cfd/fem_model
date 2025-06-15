# processing_OOP\static\operations\preparation.py

import numpy as np
from scipy.sparse import coo_matrix, issparse
from typing import List, Tuple, Union, Optional
import logging
from pathlib import Path
import os


class PrepareLocalSystem:
    """
    Validates and formats local element stiffness matrices (Ke) and force vectors (Fe)
    into the formats required by AssembleGlobalSystem:
      - Ke: scipy.sparse.coo_matrix with float64 precision
      - Fe: 1D numpy arrays with float64 precision
    """

    def __init__(
        self,
        Ke_raw: Union[List[Union[np.ndarray, coo_matrix]], np.ndarray],
        Fe_raw: Union[List[Union[np.ndarray, list]], np.ndarray],
        job_results_dir: Optional[Union[str, Path]] = None
    ):
        self.Ke_raw = self._ensure_sparse_format(Ke_raw)
        self.Fe_raw = self._ensure_flattened_vectors(Fe_raw)
        self.job_results_dir = Path(job_results_dir) if job_results_dir else None
        self.logger = self._init_logging()

    def _init_logging(self):
        logger = logging.getLogger(f"PrepareLocalSystem.{id(self)}")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        log_path = None
        if self.job_results_dir:
            log_dir = self.job_results_dir.parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "PrepareLocalSystem.log"

            try:
                file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
                file_handler.setFormatter(logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s "
                    "(Module: %(module)s, Line: %(lineno)d)"
                ))
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"⚠️ Failed to create file handler for PrepareLocalSystem class log: {e}")

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(stream_handler)

        if log_path:
            logger.debug(f"📁 Log file created at: {log_path}")

        return logger

    def _ensure_sparse_format(self, matrices) -> List[coo_matrix]:
        """Ensure all matrices are converted to COO sparse format with float64 precision."""
        if matrices is None:
            return []
        if not isinstance(matrices, list):
            matrices = list(matrices)
        return [
            coo_matrix(mat.astype(np.float64)) if not issparse(mat) else mat.astype(np.float64).tocoo()
            for mat in matrices
        ]

    def _ensure_flattened_vectors(self, vectors) -> List[np.ndarray]:
        """Ensure all vectors are 1D NumPy arrays of float64."""
        if vectors is None:
            return []
        if not isinstance(vectors, list):
            vectors = list(vectors)
        return [
            np.asarray(vec, dtype=np.float64).flatten()
            for vec in vectors
        ]

    def validate_and_format(self) -> Tuple[List[coo_matrix], List[np.ndarray]]:
        if not isinstance(self.Ke_raw, list) or not isinstance(self.Fe_raw, list):
            raise TypeError("Ke_raw and Fe_raw must be lists")

        if len(self.Ke_raw) != len(self.Fe_raw):
            raise ValueError("Ke_raw and Fe_raw must have the same number of entries")

        formatted_Ke = []
        formatted_Fe = []

        self.logger.info("🔎 Starting validation of Ke and Fe")

        for idx, (Ke_i, Fe_i) in enumerate(zip(self.Ke_raw, self.Fe_raw)):
            try:
                if not np.isfinite(Ke_i.data).all():
                    raise ValueError(f"Ke[{idx}] contains non-finite values")

                if Ke_i.shape[0] != Ke_i.shape[1]:
                    raise ValueError(f"Ke[{idx}] is not square")

                if Fe_i.ndim != 1:
                    raise ValueError(f"Fe[{idx}] must be a 1D array")
                if not np.isfinite(Fe_i).all():
                    raise ValueError(f"Fe[{idx}] contains non-finite values")
                if Fe_i.shape[0] != Ke_i.shape[0]:
                    raise ValueError(f"Fe[{idx}] shape does not match Ke[{idx}]")

                formatted_Ke.append(Ke_i)
                formatted_Fe.append(Fe_i)
                self.logger.debug(f"✅ Ke[{idx}] and Fe[{idx}] validated and formatted")

            except Exception as e:
                self.logger.error(f"❌ Validation failed at index {idx}: {e}", exc_info=True)
                raise RuntimeError(f"Validation failed at element {idx}") from e

        self.logger.info(f"✅ Successfully validated and formatted {len(formatted_Ke)} elements")
        return formatted_Ke, formatted_Fe
# pre_processing/element_library/base_logger_operator.py

import os
import numpy as np
from collections import defaultdict
from typing import Dict, Optional, ClassVar, Any
from pathvalidate import sanitize_filename
import errno

class LoggingConfigurationError(RuntimeError):
    """Custom exception for logging configuration failures"""
    pass

class LogWriteError(IOError):
    """Custom exception for log write operations"""
    pass

class PrecisionConfigurationError(ValueError):
    """Exception for invalid precision configurations"""
    pass

class BaseLoggerOperator:
    """Enterprise-grade logging operations with guaranteed numerical precision"""
    
    _CATEGORY_PATHS: ClassVar[Dict[str, str]] = {
        "stiffness": "element_stiffness_matrices",
        "force": "element_force_vectors"
    }

    def __init__(self, element_id: int, job_results_dir: str) -> None:
        """
        Initialize a precision-aware logger operator for FEM elements.
    
        Args:
            element_id: Positive integer element identifier (supports NumPy integers)
            job_results_dir: Valid output directory path
        
        Raises:
            LoggingConfigurationError: For invalid paths/permissions
        """
        if not isinstance(element_id, (int, np.integer)) or element_id < 0:
            raise ValueError(f"Invalid element_id: {element_id} (must be ≥0)")
        
        if not isinstance(job_results_dir, str) or not job_results_dir:
            raise ValueError("job_results_dir must be a non-empty string")

        self.element_id: int = int(element_id)
        self.job_results_dir: str = os.path.abspath(job_results_dir)
        self.buffers: Dict[str, list] = defaultdict(list)
        self._sanitized_id: str = sanitize_filename(str(self.element_id))

        try:
            self._validate_directory_structure()
        except OSError as e:
            raise LoggingConfigurationError(f"Filesystem error: {e}") from e

    def _validate_directory_structure(self) -> None:
        """Validate directory permissions and structure"""
        required_dirs = [
            os.path.join(self.job_results_dir, d)
            for d in self._CATEGORY_PATHS.values()
        ]
        
        for d in required_dirs:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Missing required directory: {d}")
            if not os.access(d, os.W_OK):
                raise PermissionError(f"Write access denied for: {d}")

    def _get_log_path(self, category: str) -> str:
        """
        Generate validated log file path with sanitized naming
        
        Args:
            category: Predefined logging category
            
        Returns:
            Absolute path to log file
            
        Raises:
            ValueError: For invalid category
        """
        if category not in self._CATEGORY_PATHS:
            raise ValueError(f"Invalid category: {category}. Valid options: {list(self._CATEGORY_PATHS.keys())}")
            
        return os.path.join(
            self.job_results_dir,
            self._CATEGORY_PATHS[category],
            f"{category}_element_{self._sanitized_id}.log"
        )

    def log_text(self, category: str, message: str) -> None:
        """
        Buffer text message with validation
        
        Args:
            category: Valid logging category
            message: Text to log
            
        Raises:
            ValueError: For invalid category
            TypeError: For non-string messages
        """
        if category not in self._CATEGORY_PATHS:
            raise ValueError(f"Invalid category: {category}")
        if not isinstance(message, str):
            raise TypeError(f"Message must be string, got {type(message)}")
            
        self.buffers[category].append(f"{message}\n")

    def log_matrix(
        self,
        category: str,
        matrix: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Buffer numerical matrix with precision guarantees
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError(f"Matrix must be numpy array, got {type(matrix)}")

        dtype = matrix.dtype
        if dtype not in (np.float64, np.int64):
            raise TypeError(f"Unsupported dtype {dtype}. Use float64/int64")

        # Configure display precision rules
        default_precision = 6 if dtype == np.float64 else 0  # Default to 6 decimal places
        config = {
            "precision": default_precision,
            "max_line_width": 120,
            "name": "Unnamed Matrix"
        }
    
        if metadata:
            if not isinstance(metadata, dict):
                raise TypeError("Metadata must be dictionary")
            
            # Allow precision override while maintaining storage precision
            if "precision" in metadata:
                if not (0 <= metadata["precision"] <= 15):
                    raise PrecisionConfigurationError(
                        f"Precision must be 0-15 (got {metadata['precision']})"
                    )
                config["precision"] = metadata["precision"]
            
            config.update(metadata)

        # Generate matrix string representation
        matrix_str = np.array2string(
            matrix,
            precision=config["precision"],
            suppress_small=True,
            max_line_width=config["max_line_width"],
            floatmode="maxprec" if dtype == np.float64 else "fixed"
        )
        self.buffers[category].append(
            f"=== {config['name']} ===\n{matrix_str}\n\n"
        )

    def flush(self, category: str, mode: str = "a") -> None:  # Changed default to append mode
        """Atomic write operation with append mode"""
        if mode not in {"a", "x"}:
            raise ValueError(f"Invalid mode '{mode}'. Use 'a' or 'x'")
    
        if not self.buffers[category]:
            return

        log_path = self._get_log_path(category)
    
        try:
            with open(log_path, mode, encoding="utf-8") as f:
                f.writelines(self.buffers[category])
            self.buffers[category].clear()
        
        except FileExistsError as e:
            raise LogWriteError(f"File exists (use mode='a'): {log_path}") from e
        except IOError as e:
            raise LogWriteError(f"Write failed: {e}") from e

    def flush_all(self) -> None:
        """Flush all buffers with consolidated error handling"""
        errors = []
        for category in self._CATEGORY_PATHS:
            try:
                if self.buffers[category]:
                    self.flush(category)
            except LogWriteError as e:
                errors.append(str(e))
                
        if errors:
            raise LogWriteError(f"Failed flushes:\n- " + "\n- ".join(errors))
        
    def emergency_flush(self):
        """Force write all buffers regardless of state"""
        for category in self._CATEGORY_PATHS:
            try:
                if self.buffers[category]:
                    self.flush(category, mode='a')  # Append mode in case file exists
            except Exception:
                # Last-ditch effort to save data
                log_path = self._get_log_path(category)
                with open(log_path, 'a') as f:
                    f.writelines(str(item) for item in self.buffers[category])
            self.buffers[category].clear()

    def verify_logs_exist(self) -> bool:
        """Validate existence of all expected log files"""
        return all(
            os.path.exists(self._get_log_path(category))
            for category in self._CATEGORY_PATHS
        )

    def __repr__(self) -> str:
        return (
            f"BaseLoggerOperator(element_id={self.element_id}, "
            f"categories={list(self._CATEGORY_PATHS.keys())})"
        )

    def __str__(self) -> str:
        return (
            f"Logger for Element {self.element_id} -> "
            f"{os.path.basename(self.job_results_dir)}"
        )
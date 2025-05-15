# pre_processing/element_library/base_logger_operator.py

import os
import numpy as np
from collections import defaultdict
from typing import Dict, Optional
from pathvalidate import sanitize_filename

class BaseLoggerOperator:
    """Robust logging operations for FEM analysis with validated path handling"""
    
    _CATEGORY_PATHS = {
        "stiffness": "element_stiffness_matrices",
        "force": "element_force_vectors"
    }

    def __init__(self, element_id: int, job_results_dir: str):
        """
        Initialize a validated logger operator for an element.
        
        Args:
            element_id: Unique element identifier
            job_results_dir: Base directory for all output files
        """
        if not job_results_dir:
            raise ValueError("job_results_dir must be provided")
            
        self.element_id = element_id
        self.job_results_dir = job_results_dir
        self.buffers = defaultdict(list)
        self._sanitized_id = sanitize_filename(str(element_id))
        
        self._create_category_directories()
        self._validate_directory_structure()

    def _create_category_directories(self):
        """Create all required output directories with validation"""
        try:
            for category_dir in self._CATEGORY_PATHS.values():
                full_path = os.path.join(self.job_results_dir, category_dir)
                os.makedirs(full_path, exist_ok=True)
                
                # Verify directory was actually created
                if not os.path.isdir(full_path):
                    raise RuntimeError(f"Failed to create directory: {full_path}")
        except PermissionError as e:
            raise RuntimeError(f"Permission denied creating directories: {e}") from e

    def _validate_directory_structure(self):
        """Ensure directory structure meets requirements"""
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
        Generate safe log file path with sanitized element ID.
        
        Args:
            category: One of the predefined logging categories
            
        Returns:
            Full sanitized path to the log file
        """
        if category not in self._CATEGORY_PATHS:
            raise ValueError(f"Invalid logging category: {category}")
            
        safe_filename = f"{category}_element_{self._sanitized_id}.log"
        return os.path.join(
            self.job_results_dir,
            self._CATEGORY_PATHS[category],
            safe_filename
        )

    def log_text(self, category: str, message: str):
        """Buffer text message with category validation"""
        if category not in self._CATEGORY_PATHS:
            raise ValueError(f"Invalid logging category: {category}")
        self.buffers[category].append(f"{message}\n")

    def log_matrix(self, category: str, matrix: np.ndarray, metadata: Dict = None):
        """Buffer numerical matrix with formatting controls"""
        if category not in self._CATEGORY_PATHS:
            raise ValueError(f"Invalid logging category: {category}")

        config = {
            "precision": 6,
            "max_line_width": 120,
            "name": "Unnamed Matrix"
        }
        if metadata:
            config.update(metadata)

        header = f"=== {config['name']} ===\n"
        matrix_str = np.array2string(
            matrix,
            precision=config["precision"],
            suppress_small=True,
            max_line_width=config["max_line_width"]
        )
        self.buffers[category].append(f"{header}{matrix_str}\n\n")

    def flush(self, category: str, mode: str = "x"):
        """
        Atomic write of buffered logs with conflict prevention
        
        Args:
            category: Logging category to flush
            mode: File write mode ('x' for exclusive create, 'a' for append)
        """
        if not self.buffers[category]:
            return

        log_path = self._get_log_path(category)
        
        try:
            with open(log_path, mode, encoding="utf-8") as f:
                f.writelines(self.buffers[category])
            self.buffers[category].clear()
            
            # Verify write operation
            if os.path.getsize(log_path) == 0:
                raise IOError(f"Empty log file created: {log_path}")
        except FileExistsError:
            raise RuntimeError(f"Log file already exists: {log_path}") from None
        except IOError as e:
            raise RuntimeError(f"Failed to write {log_path}: {e}") from e

    def flush_all(self):
        """Flush all buffers with validation"""
        for category in self._CATEGORY_PATHS:
            if self.buffers[category]:
                self.flush(category)
                
        # Post-flush verification
        for category in self._CATEGORY_PATHS:
            log_path = self._get_log_path(category)
            if not os.path.exists(log_path):
                raise FileNotFoundError(f"Missing expected log file: {log_path}")

    def verify_logs_exist(self) -> bool:
        """Check if all expected log files were created"""
        return all(
            os.path.exists(self._get_log_path(category))
            for category in self._CATEGORY_PATHS
        )
# processing_OOP\static\results\compute_secondary_results.py

import numpy as np
from scipy.special import roots_legendre
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
import logging

class SecondaryResultsOrchestrator:
    def __init__(
        self,
        *,
        grid_dictionary,
        element_dictionary,
        material_dictionary,
        section_dictionary,
        global_displacement: np.ndarray,
        job_results_dir: str | Path | None = None,
    ):
        self.elements = elements
        self.grid_dictionary = grid_dictionary
        self.element_dictionary = element_dictionary
        self.material_dictionary = material_dictionary
        self.section_dictionary = section_dictionary
        self.U_global = global_displacement.reshape(-1)
        self.job_results_dir = Path(job_results_dir) if job_results_dir else None
        self.logger = self._init_logging()

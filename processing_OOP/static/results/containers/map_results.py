# processing_OOP\static\results\containers\map_results.py

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

# ─────────────────────────────────────────────────────────────
# Mapping containers between DOF spaces at each solver stage
# ─────────────────────────────────────────────────────────────

@dataclass
class MapEntry:
    element_id: int
    local_dof: np.ndarray
    global_dof: np.ndarray
    fixed_flag: Optional[np.ndarray] = None
    zero_flag: Optional[np.ndarray] = None
    active_flag: Optional[np.ndarray] = None
    condensed_dof: Optional[np.ndarray] = None
    reconstructed_values: Optional[np.ndarray] = None  

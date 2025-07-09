# processing_OOP\static\results\containers\global_results.py

from dataclasses import dataclass
from typing import Optional
import numpy as np
import scipy.sparse as sp

# ─────────────────────────────────────────────────────────────
# Global-level results (system-wide matrices and vectors)
# ─────────────────────────────────────────────────────────────

@dataclass
class GlobalResults:
    # --------------------------------AssembleGlobalSystem outputs
    F_global: Optional[np.ndarray] = None
    K_global: Optional[sp.csr_matrix] = None
    # --------------------------------ModifyGlobalSystem outputs
    F_mod: Optional[np.ndarray] = None
    K_mod: Optional[sp.csr_matrix] = None
    # --------------------------------CondenseModifiedSystem outputs
    F_cond: Optional[np.ndarray] = None
    K_cond: Optional[sp.csr_matrix] = None
    # --------------------------------SolveCondensedSystem outputs
    U_cond: Optional[np.ndarray] = None
    # --------------------------------ReconstructGlobalSystem outputs
    U_global: Optional[np.ndarray] = None
    # --------------------------------PrimaryResultsOrchestrator outputs
    R_global: Optional[np.ndarray] = None
    R_residual: Optional[np.ndarray] = None
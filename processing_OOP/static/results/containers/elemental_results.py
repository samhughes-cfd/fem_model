# processing_OOP\static\results\containers\elemental_results.py

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import scipy.sparse as sp

# ─────────────────────────────────────────────────────────────
# Element-level results (stored per element)
# These results are defined in local element space, and their
# ordering matches the mesh element numbering.
# ─────────────────────────────────────────────────────────────

@dataclass
class ElementalResults:
    K_e: Optional[list[sp.csr_matrix]] = None
    # Element stiffness matrices
    # Shape per entry: (n_dofs_per_elem, n_dofs_per_elem)

    F_e: Optional[list[np.ndarray]] = None
    # Element externalforce vectors (including internal/external)
    # Shape per entry: (n_dofs_per_elem,)

    U_e: Optional[list[np.ndarray]] = None
    # Element displacement vectors
    # Shape per entry: (n_dofs_per_elem,)

    R_e: Optional[list[np.ndarray]] = None
    # Element reaction vectors (R_e = F_e - K_e @ U_e)
    # Shape per entry: (n_dofs_per_elem,)

    R_residual_e: Optional[list[np.ndarray]] = None
    # Element reaction vectors (R_e = F_e - K_e @ U_e)
    # Shape per entry: (n_dofs_per_elem,)

    total_strain_energy: Optional[list[float]] = None
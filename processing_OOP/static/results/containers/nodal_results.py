# processing_OOP\static\results\containers\nodal_results.py

# processing_OOP\static\results\containers\nodal_results.py

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import scipy.sparse as sp

# ─────────────────────────────────────────────────────────────
# Nodal-level results (interpolated field quantities at nodes)
# These results are typically obtained by extrapolating from
# integration points or directly solving for nodal quantities.
# ─────────────────────────────────────────────────────────────

@dataclass
class NodalResults:
    internal_forces: Optional[np.ndarray] = None
    # Shape: (n_nodes, 6)
    # Components: [N, Vy, Vz, T, My, Mz] at each node

    strain: Optional[np.ndarray] = None
    # Shape: (n_nodes, 6)
    # Components: [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz] or beam-specific equivalents

    stress: Optional[np.ndarray] = None
    # Shape: (n_nodes, 6)
    # Components: [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz] or beam-specific equivalents

    strain_energy_density: Optional[np.ndarray] = None
    # Shape: (n_nodes,)
    # Scalar strain energy density per node (typically in J/m³ or similar)

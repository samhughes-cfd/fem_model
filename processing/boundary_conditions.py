"""
processing/boundary_conditions.py

Applies a fixed boundary condition to the global stiffness matrix and force vector.
"""

import numpy as np
import logging
from scipy.sparse import csr_matrix

def apply_boundary_conditions(K_global, F_global):
    """
    Applies a fixed boundary condition for a cantilever beam using the **Penalty Method**.

    **Modifications:**
    1️⃣ **Zero out** the first 6 rows and columns of `K_global` to remove interactions.
    2️⃣ **Set large penalty** values on the first 6 diagonal terms of `K_global`.
    3️⃣ **Zero out** the first 6 entries of `F_global` (no external force on fixed DOFs).

    **Optimized Handling:**
    - `K_global` is already in **LIL format** (from `assembly.py`), so modifications are efficient.
    - `F_global` remains a **1D NumPy array**.
    - Converts `K_global` to **CSR format** after modification for efficient solving.

    Parameters
    ----------
    K_global : lil_matrix
        Global stiffness matrix (LIL format for efficient modification).
    F_global : ndarray
        Global force vector (1D NumPy array).

    Returns
    -------
    K_mod : csr_matrix
        Modified stiffness matrix in CSR format (optimized for solving).
    F_mod : ndarray
        Modified force vector (1D NumPy array).
    """

    logging.info("Applying cantilever beam boundary conditions.")

    # ✅ Ensure F_global is a 1D NumPy array
    F_global = np.asarray(F_global).flatten()

    # ✅ Define penalty value for enforcing zero displacements
    large_penalty = 1e12  

    # 🔹 Apply constraints to first 6 DOFs (cantilever's fixed node)
    fixed_dofs = np.arange(6)  # First 6 DOFs

    # 1️⃣ **Zero out corresponding rows and columns**
    K_global[fixed_dofs, :] = 0
    K_global[:, fixed_dofs] = 0

    # 2️⃣ **Apply large penalty value to enforce zero displacement**
    K_global.setdiag(large_penalty, k=0)  # Efficient diagonal modification

    # 3️⃣ **Ensure zero external force at fixed DOFs**
    F_global[fixed_dofs] = 0

    # ✅ Convert `K_global` to CSR format for solving
    K_mod = K_global.tocsr()

    logging.info("✅ Fixed boundary conditions applied successfully.")

    return K_mod, F_global

# processing\static\condensation.py

import os
import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# 🔧 Enhanced Logging Configuration
def configure_logging(job_results_dir, log_filename="condensation.log"):
    logger = logging.getLogger(__name__)
    logger.handlers.clear()  # Ensure no duplicate handlers
    logger.setLevel(logging.DEBUG)  # Capture all logs in the log file

    # 🔧 File Handler (Captures all logs)
    log_path = os.path.join(job_results_dir, log_filename)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] (%(funcName)s:%(lineno)d) - %(message)s"))
    logger.addHandler(file_handler)

    # ❌ Suppress console output completely
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.CRITICAL)  # Only show CRITICAL messages (which should rarely occur)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    # 🚀 Ensure logs don’t propagate to the root logger
    logger.propagate = False  # This is key to suppress terminal logging

    return logger

# 🛠️ Enhanced Condensation Process with Granular Logging
def condensation(K_mod, F_mod, fixed_dofs, job_results_dir, tol=1e-12):
    logger = configure_logging(job_results_dir)
    logger.info("🔧 Starting static condensation process.")

    num_dofs = K_mod.shape[0]
    logger.debug(f"🔍 Total DOFs in initial system: {num_dofs}")

    # Step 1: Explicit Fixed DOFs
    active_dofs = np.setdiff1d(np.arange(num_dofs), fixed_dofs)
    logger.debug(f"🚫 Explicitly Fixed DOFs: {fixed_dofs}")
    logger.debug(f"✅ Active DOFs after removing fixed: {active_dofs}")

    # Apply first stage condensation
    K_intermediate = K_mod[active_dofs][:, active_dofs]
    F_intermediate = F_mod[active_dofs]
    logger.debug(f"🟢 Intermediate K matrix shape: {K_intermediate.shape}")

    # Identify rows/columns effectively zero
    nonzero_rows = np.where(np.any(np.abs(K_intermediate.toarray()) > tol, axis=1))[0]
    fully_active_dofs = active_dofs[nonzero_rows]
    logger.info(f"🔍 DOFs removed due to zero rows/columns: {len(active_dofs) - len(fully_active_dofs)}")
    logger.debug(f"✅ Fully active DOFs post zero check: {fully_active_dofs}")

    # Final condensed matrices
    K_cond = K_intermediate[nonzero_rows][:, nonzero_rows]
    F_cond = F_intermediate[nonzero_rows]

    # Logging inactive DOFs explicitly
    inactive_dofs = np.setdiff1d(active_dofs, fully_active_dofs)
    logger.debug(f"🟠 Secondary inactive DOFs removed: {inactive_dofs}")

    # Zero diagonal checks for numerical stability
    diagonal_zeros = np.where(np.abs(K_cond.diagonal()) < tol)[0]
    if diagonal_zeros.size > 0:
        logger.warning(f"⚠️ Near-zero diagonal entries detected at condensed indices: {diagonal_zeros}")

    # Map condensed indices to original indices
    condensed_to_original_map = {i: fully_active_dofs[i] for i in range(len(fully_active_dofs))}

    # Log detailed matrix structure
    if len(fully_active_dofs) <= 100:
        df_K_cond = pd.DataFrame(
            K_cond.toarray(),
            index=[f"Cond {i} (Orig {condensed_to_original_map[i]})" for i in range(len(fully_active_dofs))],
            columns=[f"Cond {j} (Orig {condensed_to_original_map[j]})" for j in range(len(fully_active_dofs))],
        )
        df_F_cond = pd.DataFrame(
            F_cond,
            index=[f"Cond {i} (Orig {condensed_to_original_map[i]})" for i in range(len(fully_active_dofs))],
            columns=["F_cond"]
        )

        logger.debug(f"📜 Condensed K matrix:\n{df_K_cond.to_string(float_format='%.4e')}")
        logger.debug(f"📄 Condensed F vector:\n{df_F_cond.to_string(float_format='%.4e')}")

    # Log mappings clearly
    logger.info("🔗 DOF Mapping (Condensed → Original):")
    for c_idx, o_idx in condensed_to_original_map.items():
        logger.debug(f"Condensed {c_idx} → Original {o_idx}")

    logger.info("✅ Static condensation completed successfully.")
    return fully_active_dofs, inactive_dofs, K_cond, F_cond
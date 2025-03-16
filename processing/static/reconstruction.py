# processing\static\reconstruction.py

import os
import logging
import numpy as np
import pandas as pd

def configure_logging(job_results_dir, log_filename="reconstruction.log"):
    """
    📂 Configures logging for displacement reconstruction, suppressing terminal output.
    
    Parameters:
        job_results_dir (str): Directory for logs.
        log_filename (str): Name of the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(__name__)
    logger.handlers.clear()  # Remove existing handlers
    logger.setLevel(logging.DEBUG)

    # File handler (captures all logs)
    log_path = os.path.join(job_results_dir, log_filename)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    # Console handler (suppressed except CRITICAL)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.CRITICAL)  # Only shows CRITICAL logs in the terminal
    logger.addHandler(console_handler)

    return logger

def reconstruction(active_dofs, U_cond, total_dofs, job_results_dir):
    """
    🔧 Reconstructs the full displacement vector from the condensed solution.

    Parameters:
        active_dofs (np.ndarray): Indices of active DOFs.
        U_cond (np.ndarray): Condensed displacement vector.
        total_dofs (int): Total number of DOFs in the system.
        job_results_dir (str): Directory for logs.

    Returns:
        np.ndarray: Reconstructed displacement vector.
    """
    # 🔍 Configure logger
    logger = configure_logging(job_results_dir)

    logger.debug("🔧 Beginning displacement reconstruction.")
    logger.debug(f"📌 Total DOFs in system: {total_dofs}")
    logger.debug(f"📌 Condensed DOFs count: {len(active_dofs)}")
    logger.debug(f"📌 Active DOFs indices: {active_dofs}")

    # 🛠 Initialize reconstructed displacement vector
    U_global = np.zeros(total_dofs)

    logger.debug("📐 Initialized global displacement array with zeros.")

    # 🚧 Check dimension consistency
    if len(U_cond) != len(active_dofs):
        logger.error("❌ Dimension mismatch between condensed displacement vector and active DOFs.")
        raise ValueError("Dimension mismatch between U_cond and active_dofs.")

    # 🔄 Populate global displacement vector
    for i, dof in enumerate(active_dofs):
        U_global[dof] = U_cond[i]
        logger.debug(f"Mapped condensed displacement U_cond[{i}] = {U_cond[i]:.6e} to global DOF {active_dofs[i]}.")

    logger.info("✅ Successfully mapped all condensed displacements back to global vector.")

    # 📜 Detailed DOF mapping
    mapping_table = "DOF Mapping:\nIndex | Active DOF | U_cond\n"
    mapping_table += "-" * 40 + "\n"
    for i, dof in enumerate(active_dofs):
        mapping_table += f"{i:4d} → {active_dofs[i]:4d}: {U_cond[i]:.6e}\n"

    with open(os.path.join(job_results_dir, "reconstruction.log"), "a", encoding="utf-8") as f:
        f.write("\n📋 Detailed DOF Mapping:\n" + mapping_table)

    logger.debug("📝 Logged detailed DOF mapping.")

    # 📊 Structured logging of reconstructed displacements
    reconstructed_df = pd.DataFrame({
        "Global DOF": np.arange(total_dofs),
        "Displacement": U_global
    })

    with open(os.path.join(job_results_dir, "reconstruction.log"), "a", encoding="utf-8") as f:
        f.write("\n📊 Full reconstructed displacement vector:\n" + reconstructed_df.to_string(float_format="%.6e"))

    logger.info("📘 Reconstruction log updated with structured displacement data.")

    # 🎯 Return reconstructed displacement vector
    return U_global
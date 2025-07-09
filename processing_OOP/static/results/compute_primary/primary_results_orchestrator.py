# processing_OOP\static\results\compute_primary\primary_results_orchestrator.py

import numpy as np
import scipy.sparse as sp
from pathlib import Path
import logging

from .compute_reaction_force import ComputeReactionForce
from .compute_residual import ComputeResidual


class PrimaryResultsOrchestrator:
    def __init__(
        self,
        *,
        K_global: sp.csr_matrix,
        F_global: np.ndarray,
        U_global: np.ndarray,
        fixed_dofs: np.ndarray,
        job_results_dir: str | Path | None = None,
    ):
        self.K_global = K_global
        self.F_global = F_global.reshape(-1)  # Ensure 1D
        self.U_global = U_global.reshape(-1)  # Ensure 1D
        self.fixed_dofs = np.asarray(fixed_dofs)
        self.job_results_dir = Path(job_results_dir) if job_results_dir else None
        self.logger = self._init_logging()

    def compute(self) -> tuple[np.ndarray, np.ndarray]:
        self._basic_sanity_checks()

        R_global = ComputeReactionForce(
            K_global=self.K_global,
            F_global=self.F_global,
            U_global=self.U_global,
            fixed_dofs=self.fixed_dofs
        ).compute()

        R_residual = ComputeResidual(
            K_global=self.K_global,
            F_global=self.F_global,
            U_global=self.U_global
        ).compute()

        self.logger.info("✅ Reactions evaluated & masked")
        self.logger.info("✅ Residual computed")

        #if self.job_results_dir:
            #np.savetxt(self.job_results_dir / "R_global.csv", R_global, delimiter=",")
            #np.savetxt(self.job_results_dir / "R_residual.csv", R_residual, delimiter=",")
            #self.logger.info("💾 Results written to disk")

        return R_global, R_residual

    def _basic_sanity_checks(self) -> None:
        n = self.K_global.shape[0]
        if self.U_global.shape != (n,):
            raise ValueError(f"U_global shape {self.U_global.shape} does not match K_global")
        if self.F_global.shape != (n,):
            raise ValueError(f"F_global shape {self.F_global.shape} does not match K_global")

    def _init_logging(self):
        logger = logging.getLogger(f"PrimaryResultsOrchestrator.{id(self)}")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        log_path = None
        if self.job_results_dir:
            logs_dir = self.job_results_dir.parent / "logs"  # ✅ Store logs alongside primary_results
            logs_dir.mkdir(parents=True, exist_ok=True)

            log_path = logs_dir / "PrimaryResultsOrchestrator.log"
            try:
                file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
                file_handler.setFormatter(logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s "
                    "(Module: %(module)s, Line: %(lineno)d)"
                ))
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"⚠️ Failed to create file handler for PrimaryResultsOrchestrator class log: {e}")

        # Console output (INFO level and above)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(stream_handler)

        if log_path:
            logger.debug(f"📁 Log file created at: {log_path}")

        return logger
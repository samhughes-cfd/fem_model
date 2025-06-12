# processing_OOP\static\operations\disassembly.py

import os
import time
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Sequence
from functools import partial
from multiprocessing import Pool, cpu_count

from scipy.sparse import csr_matrix, issparse


# --------------------------------------------------------------------------- #
#  MAIN CLASS
# --------------------------------------------------------------------------- #
class DisassembleGlobalSystem:
    """
    Element-wise ‘reverse assembly’ with validation, (optional) parallelism,
    and detailed diagnostics.

    Parameters
    ----------
    elements : Sequence[object]
        Same element objects that were passed to `AssembleGlobalSystem`.
        **Must provide** a callable `assemble_global_dof_indices(element_id)`
        which returns the global DOF mapping for that element.
    K_mod : csr_matrix
        Boundary-conditioned global stiffness matrix.  *Not* the condensed one —
        use the matrix *before* static condensation.
    F_mod, U_global, R_global : np.ndarray
        Force, displacement, and reaction vectors of **length = total DOFs**.
    job_results_dir : str | Path, optional
        Folder for `disassembly.log` and diagnostics.
    parallel : bool, default False
        Use a `multiprocessing.Pool` to extract element data in parallel.
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        elements: Sequence[object],
        K_mod: csr_matrix,
        F_mod: np.ndarray,
        U_global: np.ndarray,
        R_global: np.ndarray,
        job_results_dir: Optional[str] = None,
        parallel: bool = False,
    ):
        self.elements: Sequence[object] = elements
        self.K_mod: csr_matrix = K_mod
        self.F_mod: np.ndarray = np.asarray(F_mod, dtype=np.float64).ravel()
        self.U_global: np.ndarray = np.asarray(U_global, dtype=np.float64).ravel()
        self.R_global: np.ndarray = np.asarray(R_global, dtype=np.float64).ravel()
        self.job_results_dir: Optional[str] = (
            str(job_results_dir) if job_results_dir else None
        )
        self.parallel: bool = bool(parallel)

        self._init_logging()
        self._validate_inputs()
        self._compute_dof_mappings()

        # outputs
        self.K_e_mod: List[csr_matrix] = []
        self.F_e_mod: List[np.ndarray] = []
        self.U_e: List[np.ndarray] = []
        self.R_e: List[np.ndarray] = []

    # ------------------------------------------------------------------ #
    #                P U B L I C   I N T E R F A C E
    # ------------------------------------------------------------------ #
    def disassemble(
        self,
    ) -> Tuple[List[csr_matrix], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Extract element-wise quantities.

        Returns
        -------
        K_e_mod, F_e_mod, U_e, R_e : lists
            Lists are ordered exactly like `elements`.
        """
        t0 = time.perf_counter()
        self.logger.info("🔧 Starting element disassembly …")

        extractor = partial(self._extract_one_element, K=self.K_mod)

        if self.parallel:
            with Pool(cpu_count()) as pool:
                results = pool.starmap(
                    extractor,
                    zip(
                        self.dof_mappings,
                        self.F_mod,
                        self.U_global,
                        self.R_global,  # these are scalars, but that‘s okay
                    ),
                )
        else:
            results = [
                extractor(
                    dof_map,
                    self.F_mod,  # same global vectors each time – pass once
                    self.U_global,
                    self.R_global,
                )
                for dof_map in self.dof_mappings
            ]

        # unpack
        (
            self.K_e_mod,
            self.F_e_mod,
            self.U_e,
            self.R_e,
        ) = map(list, zip(*results))

        self.disassembly_time = time.perf_counter() - t0
        self._log_performance()
        self.logger.info("✅ Element disassembly complete.")
        return self.K_e_mod, self.F_e_mod, self.U_e, self.R_e

    # ------------------------------------------------------------------ #
    #                P R I V A T E   H E L P E R S
    # ------------------------------------------------------------------ #
    def _init_logging(self):
        """Configure per-instance logger."""
        self.logger = logging.getLogger(f"DisassemblySystem.{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # file output
        if self.job_results_dir:
            os.makedirs(self.job_results_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(self.job_results_dir, "disassembly.log"), mode="w")
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s "
                    "(Module: %(module)s, Line: %(lineno)d)"
                )
            )
            self.logger.addHandler(fh)

        # console (warnings+)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self.logger.addHandler(ch)

    # ------------------------------------------------------------------ #
    def _validate_inputs(self):
        """Basic type/shape checks."""
        if not issparse(self.K_mod):
            raise TypeError("K_mod must be a scipy sparse matrix (CSR).")

        n = self.K_mod.shape[0]
        for name, vec in {
            "F_mod": self.F_mod,
            "U_global": self.U_global,
            "R_global": self.R_global,
        }.items():
            if vec.ndim != 1:
                raise ValueError(f"{name} must be a 1-D array")
            if vec.size != n:
                raise ValueError(
                    f"{name}.size ({vec.size}) does not match K_mod.shape[0] ({n})"
                )

        self.logger.debug("Input validation passed.")

    # ------------------------------------------------------------------ #
    def _compute_dof_mappings(self):
        """Reuse each element’s existing global DOF mapping."""
        self.dof_mappings: List[np.ndarray] = []
        errs = []
        for idx, elem in enumerate(self.elements):
            try:
                gmap = np.asarray(elem.assemble_global_dof_indices(elem.element_id)).ravel()
                if gmap.size == 0 or gmap.min() < 0 or gmap.max() >= self.K_mod.shape[0]:
                    raise ValueError("invalid DOF indices")
                self.dof_mappings.append(gmap.astype(np.int32))
            except Exception as exc:
                errs.append((idx, exc))
                self.logger.error("Element %s DOF mapping failed: %s", idx, exc)

        if errs:
            raise RuntimeError(f"{len(errs)} elements have invalid DOF mappings")

        self.logger.debug("Collected DOF mappings for %d elements", len(self.elements))

    # ------------------------------------------------------------------ #
    def _extract_one_element(
        self,
        dof_map: np.ndarray,
        F: np.ndarray,
        U: np.ndarray,
        R: np.ndarray,
        *,
        K: csr_matrix,
    ):
        """
        Slice out element-wise sub-matrices/vectors.

        Returns
        -------
        (K_e_mod, F_e_mod, U_e, R_e)
        """
        K_e = K[dof_map][:, dof_map].tocsr()
        F_e = F[dof_map].copy()
        U_e = U[dof_map].copy()
        R_e = R[dof_map].copy()
        return K_e, F_e, U_e, R_e

    # ------------------------------------------------------------------ #
    def _log_performance(self):
        """Write a short performance summary into the log."""
        stats = [
            f"Elements processed : {len(self.elements)}",
            f"Disassembly time   : {self.disassembly_time:.2f} s",
            f"Rate               : {len(self.elements)/self.disassembly_time:.1f} elem/s",
        ]
        self.logger.info("📊 Disassembly stats:\n  " + "\n  ".join(stats))


# --------------------------------------------------------------------------- #
#  Convenience function (optional)
# --------------------------------------------------------------------------- #
def disassemble_global_matrices(
    elements: Sequence[object],
    K_mod: csr_matrix,
    F_mod: np.ndarray,
    U_global: np.ndarray,
    R_global: np.ndarray,
    *,
    job_results_dir: Optional[str] = None,
    parallel: bool = False,
):
    """
    Functional façade so existing code that still calls the *old* function name
    keeps working.  Internally it just instantiates the class above.

    Returns
    -------
    (K_e_mod, F_e_mod, U_e, R_e) – four lists in element order.
    """
    dis = DisassembleGlobalSystem(
        elements,
        K_mod,
        F_mod,
        U_global,
        R_global,
        job_results_dir=job_results_dir,
        parallel=parallel,
    )
    return dis.disassemble()
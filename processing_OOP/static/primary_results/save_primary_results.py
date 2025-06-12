# processing_OOP\static\primary_results\save_primary_results.py

import logging, json, os
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import scipy.sparse as sp
from dataclasses import asdict

# --------------------------------------------------------------------------- #
#  P U B L I C   C L A S S
# --------------------------------------------------------------------------- #
class SavePrimaryResults:
    """
    Responsible *only* for persisting primary results to disk.

    Parameters
    ----------
    job_name : str
        Used in file names.
    start_timestamp : str
        Constant timestamp captured when the runner started
        (ISO-8601 string recommended).
    output_root : str | Path
        Directory where a sub-dir ``primary_results`` will be created.
    primary_set : PrimaryResultSet
        Global-level payload returned by ``ComputePrimaryResults.compute()``.
    element_records : dict[str, List[Any]]
        Mapping *key → list per element* returned by ``DisassembleGlobalSystem``.
        Each list **must** be ordered exactly like the original ``elements`` list.
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        job_name: str,
        start_timestamp: str,
        output_root: str | Path,
        *,
        primary_set,
        element_records: Dict[str, List[Any]] | None = None,
        logger: logging.Logger | None = None,
    ):
        from processing_OOP.static.primary_results.compute_primary_results import PrimaryResultSet

        if not isinstance(primary_set, PrimaryResultSet):
            raise TypeError("primary_set must be a PrimaryResultSet instance")

        self.job_name   = job_name
        self.timestamp  = start_timestamp
        self.root       = Path(output_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.out_dir    = self.root / "primary_results"
        self.out_dir.mkdir(exist_ok=True)

        self.global_set = primary_set
        self.elem_rec   = element_records or {}

        self.log = logger or logging.getLogger("SavePrimaryResults")
        if not self.log.handlers:
            self.log.addHandler(logging.StreamHandler())
        self.log.setLevel(logging.INFO)

    # ------------------------------------------------------------------ #
    #  P U B L I C   A P I
    # ------------------------------------------------------------------ #
    def write_all(self):
        self._write_global()
        self._write_element()
        self._write_manifest()
        self.log.info("✅ Primary results written to «%s»", self.out_dir)

    # ------------------------------------------------------------------ #
    #  I N T E R N A L   H E L P E R S
    # ------------------------------------------------------------------ #
    # ----- GLOBAL ----------------------------------------------------- #
    def _write_global(self):
        g = self.global_set

        # sparse matrices -> .npz (CSR)
        sp.save_npz(self.out_dir / self._fname("global", "K_global", "npz"), g.K_global)
        sp.save_npz(self.out_dir / self._fname("global", "K_mod",    "npz"), g.K_mod)

        # dense arrays -> compressed npz bundle
        np.savez_compressed(
            self.out_dir / self._fname("global", "dense_arrays", "npz"),
            F_global = g.F_global,
            F_mod    = g.F_mod,
            U_global = g.U_global,
            R_global = g.R_global,
        )
        # element DOF maps -> json (small, human-readable)
        with open(self.out_dir / self._fname("global", "element_dof_maps", "json"), "w") as fh:
            json.dump([m.tolist() for m in g.element_dof_maps], fh)

        self.log.debug("Global payload written")

    # ----- ELEMENT ---------------------------------------------------- #
    def _write_element(self):
        if not self.elem_rec:
            self.log.warning("No element-wise records supplied; skipping element write")
            return

        n_elem = len(next(iter(self.elem_rec.values())))
        # open a single file per key; append element blocks
        handles = {
            key: open(self.out_dir / self._fname("element", key, "txt"), "w")
            for key in self.elem_rec.keys()
        }
        try:
            for idx in range(n_elem):
                for key, fh in handles.items():
                    fh.write(f"\n# Element ID: {idx}\n")
                    val = self.elem_rec[key][idx]
                    self._write_value(fh, val)
        finally:
            for fh in handles.values():
                fh.close()

        self.log.debug("Element payload (%d elements) written", n_elem)

    # ----- MANIFEST --------------------------------------------------- #
    def _write_manifest(self):
        manifest = {
            "job":        self.job_name,
            "timestamp":  self.timestamp,
            "files":      sorted(str(p.name) for p in self.out_dir.iterdir()),
        }
        with open(self.out_dir / "manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)

    # ----- UTILITIES -------------------------------------------------- #
    def _fname(self, scale: str, key: str, ext: str) -> str:
        return f"{self.job_name}_static_{scale}_{key}_{self.timestamp}.{ext}"

    @staticmethod
    def _write_value(fh, obj):
        """Write ndarray or sparse or generic num to current open filehandle."""
        if isinstance(obj, np.ndarray):
            np.savetxt(fh, obj.reshape(1, -1) if obj.ndim == 1 else obj, fmt="%.6e")
        elif sp.issparse(obj):
            coo = obj.tocoo()
            for r, c, v in zip(coo.row, coo.col, coo.data):
                fh.write(f"{r}, {c}, {v:.6e}\n")
        else:  # scalar
            fh.write(f"{obj}\n")
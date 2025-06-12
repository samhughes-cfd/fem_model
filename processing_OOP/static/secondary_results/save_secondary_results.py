# 

"""
Disk-writer for derived / secondary FEM quantities.

File layout
-----------
<job_root>/
└─ secondary_results/
   ├─ gauss/
   │   ├─ element_000.npz   (# → one per element)
   │   └─ ...
   ├─ nodal_energy.npy
   └─ manifest.json          (index of everything above)
"""
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from dataclasses import asdict
from processing_OOP.static.secondary_results.compute_secondary_results import SecondaryResultSet


class SaveSecondaryResults:
    """
    Pure I/O —no maths.  Writes `SecondaryResultSet` to disk.

    Parameters
    ----------
    job_name, start_timestamp, output_root
        Same meaning and naming convention as in *SavePrimaryResults*.
    secondary_set : SecondaryResultSet
        Output of `ComputeSecondaryResults.compute()`.
    """

    def __init__(
        self,
        job_name: str,
        start_timestamp: str,
        output_root: str | Path,
        *,
        secondary_set: SecondaryResultSet,
        logger: logging.Logger | None = None,
    ):
        if not isinstance(secondary_set, SecondaryResultSet):
            raise TypeError("secondary_set must be a SecondaryResultSet instance")

        self.job_name   = job_name
        self.ts         = start_timestamp
        self.out_root   = Path(output_root).expanduser().resolve()
        self.out_root.mkdir(parents=True, exist_ok=True)

        self.out_dir    = self.out_root / "secondary_results"
        self.out_dir.mkdir(exist_ok=True)
        self.gauss_dir  = self.out_dir / "gauss"
        self.gauss_dir.mkdir(exist_ok=True)

        self.sec = secondary_set

        self.log = logger or logging.getLogger("SaveSecondaryResults")
        if not self.log.handlers:
            self.log.addHandler(logging.StreamHandler())
        self.log.setLevel(logging.INFO)

    # ------------------------------------------------------------------ #
    def write_all(self):
        self._write_gauss()
        self._write_nodal_energy()
        self._write_manifest()
        self.log.info("✅ Secondary results written to «%s»", self.out_dir)

    # ------------------------------------------------------------------ #
    #              I N T E R N A L   H E L P E R S
    # ------------------------------------------------------------------ #
    def _write_gauss(self):
        for elem_id, gp_list in self.sec.gauss_data.items():
            # serialise variable-length lists field-wise for NumPy
            np.savez(
                self.gauss_dir / f"element_{elem_id:04d}.npz",
                xi      = np.array([gp.xi     for gp in gp_list]),
                x       = np.array([gp.x      for gp in gp_list]),
                stress  = np.stack([gp.stress for gp in gp_list]),
                strain  = np.stack([gp.strain for gp in gp_list]),
                shear   = np.array([gp.shear  for gp in gp_list]),
                moment  = np.array([gp.moment for gp in gp_list]),
            )
        self.log.debug("Gauss-point payload (%d elements) written", len(self.sec.gauss_data))

    def _write_nodal_energy(self):
        np.save(self.out_dir / "nodal_energy.npy", self.sec.nodal_energy)
        self.log.debug("Nodal-energy vector written")

    def _write_manifest(self):
        files = [str(p.relative_to(self.out_dir)) for p in self.out_dir.rglob("*") if p.is_file()]
        manifest = {
            "job":        self.job_name,
            "timestamp":  self.ts,
            "files":      sorted(files),
        }
        with open(self.out_dir / "manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)
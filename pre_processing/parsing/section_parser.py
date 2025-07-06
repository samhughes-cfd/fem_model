# pre_processing\parsing\section_parser.py

import os
from typing import Dict, List
import numpy as np
import numpy.typing as npt
import pandas as pd


class SectionParser:
    """
    Parses a [Section] block and returns

        {
            "section_dictionary": {
                "element_id": np.ndarray[int64],
                "A":          np.ndarray[float64],
                "I_x":        np.ndarray[float64],
                "I_y":        np.ndarray[float64],
                "I_z":        np.ndarray[float64],
                "J_t":        np.ndarray[float64],
            }
        }
    """

    def __init__(self, filepath: str, job_results_dir: str) -> None:
        self.filepath: str = filepath
        self.job_results_dir: str = job_results_dir
        self.output_filename: str = "section_properties_parsed.csv"
        self.expected_subheader: List[str] = [
            "[element_id]", "[A]", "[I_x]", "[I_y]", "[I_z]", "[J_t]"
        ]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _assert_exact_subheader(line: str, expected: List[str]) -> None:
        tokens = [tok.lower() for tok in line.split()]
        if tokens != [hdr.lower() for hdr in expected]:
            raise ValueError(
                f"Sub-header must match (case-insensitive): {' '.join(expected)}"
            )

    # ------------------------------------------------------------------ #
    def parse(self) -> Dict[str, Dict[str, npt.NDArray]]:
        lists: Dict[str, List[float]] = {
            "element_id": [],
            "A":  [],
            "I_x": [],
            "I_y": [],
            "I_z": [],
            "J_t": [],
        }
        seen_ids: set[int] = set()

        # ---- Read & clean ---------------------------------------------- #
        with open(self.filepath, "r", encoding="utf-8") as fh:
            lines = [
                ln.strip()
                for ln in fh
                if ln.strip() and not ln.lstrip().startswith("#")
            ]

        # Locate [Section]
        try:
            start_idx = next(i for i, ln in enumerate(lines) if ln.lower() == "[section]")
        except StopIteration:
            raise ValueError("Missing [Section] section header.")

        # Validate sub-header
        self._assert_exact_subheader(lines[start_idx + 1], self.expected_subheader)

        # ---- Parse rows ------------------------------------------------- #
        for ln in lines[start_idx + 2:]:
            parts = ln.split()
            if len(parts) != 6:
                raise ValueError(f"Malformed section row: {ln!r}")

            try:
                eid = int(parts[0])
                A, Ix, Iy, Iz, Jt = map(float, parts[1:6])
            except ValueError as exc:
                raise TypeError(f"Bad data types in line {ln!r} → {exc}") from exc

            if eid in seen_ids:
                raise ValueError(f"Duplicate element_id: {eid}")
            seen_ids.add(eid)

            lists["element_id"].append(eid)
            lists["A"].append(A)
            lists["I_x"].append(Ix)
            lists["I_y"].append(Iy)
            lists["I_z"].append(Iz)
            lists["J_t"].append(Jt)

        # ---- Convert to NumPy arrays ----------------------------------- #
        parsed: Dict[str, npt.NDArray] = {
            k: np.asarray(v, dtype=(np.int64 if k == "element_id" else np.float64))
            for k, v in lists.items()
        }

        # ---- Uniform return structure ---------------------------------- #
        return {"section_dictionary": parsed}
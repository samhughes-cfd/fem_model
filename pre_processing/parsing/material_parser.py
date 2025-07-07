# pre_processing\parsing\material_parser.py

import os
from typing import Dict, List
import numpy as np
import numpy.typing as npt


class MaterialParser:
    """
    Parses a [Material] section and returns

        {
            "material_dictionary": {
                "element_id": np.ndarray[int64],
                "E":          np.ndarray[float64],
                "G":          np.ndarray[float64],
                "nu":         np.ndarray[float64],
                "rho":        np.ndarray[float64],
            }
        }
    """

    def __init__(self, filepath: str, job_results_dir: str) -> None:
        self.filepath: str = filepath
        self.job_results_dir: str = job_results_dir
        self.output_filename: str = "material_properties_parsed.csv"
        self.expected_subheader: List[str] = [
            "[element_id]", "[E]", "[G]", "[nu]", "[rho]"
        ]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _assert_exact_subheader(line: str, expected: List[str]) -> None:
        tokens = [tok.lower() for tok in line.split()]
        if tokens != [hdr.lower() for hdr in expected]:
            raise ValueError(
                f"Sub-header must match (case-insensitive): {' '.join(expected)}"
            )

    @staticmethod
    def _preprocess_lines(filepath: str) -> List[str]:
        """
        Reads a file and returns a list of stripped lines, skipping empty lines
        and lines that start with '#' (comments).
        """
        with open(filepath, "r", encoding="utf-8") as fh:
            return [
                ln.strip()
                for ln in fh
                if ln.strip() and not ln.lstrip().startswith("#")
            ]

    # ------------------------------------------------------------------ #
    def parse(self) -> Dict[str, Dict[str, npt.NDArray]]:
        data_lists: Dict[str, List[float]] = {
            "element_id": [],
            "E":  [],
            "G":  [],
            "nu": [],
            "rho": [],
        }
        seen_ids: set[int] = set()

        # ---- Read & clean ---------------------------------------------- #
        lines = self._preprocess_lines(self.filepath)

        # Locate [Material]
        try:
            start_idx = next(
                i for i, ln in enumerate(lines) if ln.lower() == "[material]"
            )
        except StopIteration:
            raise ValueError("Missing [Material] section header.")

        # Validate sub-header
        self._assert_exact_subheader(lines[start_idx + 1], self.expected_subheader)

        # ---- Parse rows ------------------------------------------------- #
        for ln in lines[start_idx + 2:]:
            parts = ln.split()
            if len(parts) != 5:
                raise ValueError(f"Malformed material row: {ln!r}")

            try:
                eid = int(parts[0])
                E, G, nu, rho = map(float, parts[1:5])
            except ValueError as exc:
                raise TypeError(f"Bad data types in line {ln!r} → {exc}") from exc

            if eid in seen_ids:
                raise ValueError(f"Duplicate element_id: {eid}")
            seen_ids.add(eid)

            data_lists["element_id"].append(eid)
            data_lists["E"].append(E)
            data_lists["G"].append(G)
            data_lists["nu"].append(nu)
            data_lists["rho"].append(rho)

        # ---- Convert to NumPy arrays ----------------------------------- #
        parsed: Dict[str, npt.NDArray] = {
            k: np.asarray(v, dtype=(np.int64 if k == "element_id" else np.float64))
            for k, v in data_lists.items()
        }

        # ---- Uniform return structure ---------------------------------- #
        return {"material_dictionary": parsed}
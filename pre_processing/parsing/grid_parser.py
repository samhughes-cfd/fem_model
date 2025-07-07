# pre_processing\parsing\grid_parser.py

import os
from typing import Dict, List
import numpy as np
import numpy.typing as npt


class GridParser:
    """
    Parses a [Grid] section and returns::

        {
            "grid_dictionary": {
                "ids":         np.ndarray[int64],
                "coordinates": np.ndarray[float64]  # shape (N, 3)
            }
        }
    """

    def __init__(self, filepath: str, job_results_dir: str) -> None:
        self.filepath: str = filepath
        self.job_results_dir: str = job_results_dir
        self.output_filename: str = "grid_parsed.csv"
        self.expected_subheader: List[str] = ["[node_id]", "[x]", "[y]", "[z]"]

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
        node_ids: List[int] = []
        node_coords: List[List[float]] = []
        seen_ids: set[int] = set()

        # ---- Read & clean ------------------------------------------------ #
        lines = self._preprocess_lines(self.filepath)

        # Locate [Grid]
        try:
            start_idx = next(i for i, ln in enumerate(lines) if ln.lower() == "[grid]")
        except StopIteration:
            raise ValueError("Missing [Grid] section header.")

        # Validate sub-header
        self._assert_exact_subheader(lines[start_idx + 1], self.expected_subheader)

        # ---- Parse rows --------------------------------------------------- #
        for ln in lines[start_idx + 2:]:
            parts = ln.split()
            if len(parts) != 4:
                raise ValueError(f"Malformed grid row: {ln!r}")

            try:
                nid = int(parts[0])
                x, y, z = map(float, parts[1:4])
            except ValueError as exc:
                raise TypeError(f"Bad data types in line {ln!r} → {exc}") from exc

            if nid in seen_ids:
                raise ValueError(f"Duplicate node_id: {nid}")
            seen_ids.add(nid)

            node_ids.append(nid)
            node_coords.append([x, y, z])

        # ---- Convert to NumPy ------------------------------------------- #
        ids_arr = np.asarray(node_ids, dtype=np.int64)
        coords_arr = np.asarray(node_coords, dtype=np.float64)

        # ---- Return uniform structure ------------------------------------ #
        return {
            "grid_dictionary": {
                "ids": ids_arr,
                "coordinates": coords_arr,
            }
        }
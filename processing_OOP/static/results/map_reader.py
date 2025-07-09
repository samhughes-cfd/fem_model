# processing_OOP/static/utility/map_reader.py

import numpy as np
from typing import List, Union, Optional
from processing_OOP.static.results.containers.map_results import MapEntry


class MapReader:
    def __init__(self, map_entries: List[MapEntry]):
        """
        Reader class for structured DOF maps using MapEntry dataclass.

        Parameters
        ----------
        map_entries : List[MapEntry]
            List of DOF mapping entries (e.g., assembly_map, condensation_map).
        """
        self.map_entries = map_entries

    def extract_unique(
        self,
        field: str = "global_dof",
        ignore_values: Optional[Union[List[int], List[float], List[str]]] = None,
    ) -> np.ndarray:
        """
        Extract sorted unique DOFs or values from the specified MapEntry field.

        Parameters
        ----------
        field : str
            Which MapEntry field to extract (e.g., 'global_dof', 'condensed_dof', 'reconstructed_values').
        ignore_values : list, optional
            Values to exclude (e.g., [-1] to skip invalid DOFs or reconstructions).

        Returns
        -------
        np.ndarray
            Sorted array of unique DOFs or values.
        """
        values = []

        for entry in self.map_entries:
            array = getattr(entry, field, None)
            if array is None:
                continue

            array = np.asarray(array)

            if array.dtype.kind == "U":
                array = array[array != ""]  # Remove empty strings

            if ignore_values is not None:
                array = array[~np.isin(array, ignore_values)]

            values.append(array)

        if not values:
            return np.array([], dtype=int)

        flat = np.concatenate(values)
        dtype = float if "reconstructed" in field or flat.dtype.kind == "f" else int
        return np.unique(flat).astype(dtype)

    def resolve(
        self,
        field: str = "global_dof",
        ignore_values: Optional[Union[List[int], List[float], List[str]]] = None,
    ) -> np.ndarray:
        """
        Wrapper for extract_unique — resolves all entries for a specific field.

        Parameters
        ----------
        field : str
            Which MapEntry field to resolve.
        ignore_values : list, optional
            Values to exclude.

        Returns
        -------
        np.ndarray
            Sorted resolved DOF indices or values.
        """
        return self.extract_unique(field=field, ignore_values=ignore_values)
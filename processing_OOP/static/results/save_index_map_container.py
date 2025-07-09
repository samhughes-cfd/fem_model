# processing_OOP\static\results\save_index_map_container.py

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def clean_array(array):
    """Convert NumPy or array-like to list of Python ints for clean string conversion."""
    return [int(x) for x in array]

class SaveIndexMaps:
    def __init__(self, index_map_set, save_dir):
        """
        Save index maps (DOF mappings) to detailed CSV files.

        Parameters
        ----------
        index_map_set : IndexMapSet
            Contains assembly, modification, condensation, reconstruction maps.
        save_dir : str or Path
            Target directory (e.g., self.maps_dir).
        """
        self.maps = index_map_set
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        self._save_assembly_map()
        self._save_modification_map()
        self._save_condensation_map()
        self._save_reconstruction_map()
        logger.info("📁 Index maps successfully written to disk")

    def _save_assembly_map(self):
        assembly_map = self.maps.assembly_map
        if not assembly_map:
            return

        records = []
        for i, entry in enumerate(assembly_map):
            records.append({
                "Element ID": i,
                "Local DOF": str(clean_array(entry.local_dof)),
                "Global DOF": str(clean_array(entry.global_dof)),
            })

        pd.DataFrame(records).to_csv(self.save_dir / "assembly_map.csv", index=False)

    def _save_modification_map(self):
        mod = self.maps.modification_map
        if not mod:
            return

        records = []
        for i, entry in enumerate(mod):
            records.append({
                "Element ID": i,
                "Local DOF": str(clean_array(entry.local_dof)),
                "Global DOF": str(clean_array(entry.global_dof)),
                "Fixed(1)/Free(0) Flag": str(clean_array(entry.fixed_flag)),
            })

        pd.DataFrame(records).to_csv(self.save_dir / "modification_map.csv", index=False)

    def _save_condensation_map(self):
        cond = self.maps.condensation_map
        if not cond:
            return

        records = []
        for i, entry in enumerate(cond):
            records.append({
                "Element ID": i,
                "Local DOF": str(clean_array(entry.local_dof)),
                "Global DOF": str(clean_array(entry.global_dof)),
                "Fixed(1)/Free(0) Flag": str(clean_array(entry.fixed_flag)),
                "Zero(1)/Non-zero(0) Flag": str(clean_array(entry.zero_flag)),
                "Active(1)/Inactive(0) Flag": str(clean_array(entry.active_flag)),
                "Condensed DOF": str(clean_array(entry.condensed_dof)),
            })

        pd.DataFrame(records).to_csv(self.save_dir / "condensation_map.csv", index=False)

    def _save_reconstruction_map(self):
        recon = self.maps.reconstruction_map
        if not recon:
            return

        records = []
        for i, entry in enumerate(recon):
            records.append({
                "Element ID": i,
                "Local DOF": str(clean_array(entry.local_dof)),
                "Global DOF": str(clean_array(entry.global_dof)),
                "Fixed(1)/Free(0) Flag": str(clean_array(entry.fixed_flag)),
                "Zero(1)/Non-zero(0) Flag": str(clean_array(entry.zero_flag)),
                "Active(1)/Inactive(0) Flag": str(clean_array(entry.active_flag)),
                "Condensed DOF": str(clean_array(entry.condensed_dof)),
                "Reconstructed Global DOF": str(clean_array(entry.reconstructed_values)),
            })

        pd.DataFrame(records).to_csv(self.save_dir / "reconstruction_map.csv", index=False)
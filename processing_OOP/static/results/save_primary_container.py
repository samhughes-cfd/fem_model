# processing_OOP\static\results\save_primary_container.py

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from scipy.sparse import issparse
from processing_OOP.static.results.map_reader import MapReader


logger = logging.getLogger(__name__)

class SavePrimaryResults:
    def __init__(self, primary_results_set, index_map_set, save_dir):
        """
        Save primary results (global and element-level) to CSV files.

        Parameters
        ----------
        primary_results_set : PrimaryResultSet
            Container holding global and element-level result objects.
        index_map_set : IndexMapSet
            Container holding all DOF index mappings.
        save_dir : str or Path
            Base directory where results should be saved.
        """
        self.results = primary_results_set
        self.maps = index_map_set
        self.save_dir = Path(save_dir)
        self.global_dir = self.save_dir / "global"
        self.elemental_dir = self.save_dir / "elemental"

        # Elemental result subdirectories
        self.element_stiffness_dir = self.elemental_dir / "element_stiffness"
        self.external_force_dir    = self.elemental_dir / "external_force"
        self.deformation_dir       = self.elemental_dir / "deformation"
        self.reaction_force_dir    = self.elemental_dir / "reaction_force"
        self.residual_reaction_dir = self.elemental_dir / "residual"

        # Create all directories
        for directory in [
            self.global_dir,
            self.element_stiffness_dir,
            self.external_force_dir,
            self.deformation_dir,
            self.reaction_force_dir,
            self.residual_reaction_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Main entry point to save all primary results."""
        self._save_global_results()
        self._save_element_results()
        logger.info("📁 Primary results successfully written to disk")

    # ──────────────────────────────────────────────────────────────
    # GLOBAL RESULTS
    # ──────────────────────────────────────────────────────────────

    def _save_global_results(self):
        gr = self.results.global_results

        # Save stiffness matrices (COO format)
        self._save_global_matrix(gr.K_global, self.maps.assembly_map, self.global_dir / "K_global.csv")
        self._save_global_matrix(gr.K_mod, self.maps.modification_map, self.global_dir / "K_mod.csv")
        self._save_global_matrix(gr.K_cond, self.maps.condensation_map, self.global_dir / "K_cond.csv")

        # Save vectors with corresponding DOF index
        self._save_global_vector(gr.F_global, self.maps.assembly_map, self.global_dir / "F_global.csv")
        self._save_global_vector(gr.F_mod, self.maps.modification_map, self.global_dir / "F_mod.csv")
        self._save_global_vector(gr.F_cond, self.maps.condensation_map, self.global_dir / "F_cond.csv")

        self._save_global_vector(gr.U_cond, self.maps.condensation_map, self.global_dir / "U_cond.csv")
        self._save_global_vector(gr.U_global, self.maps.assembly_map, self.global_dir / "U_global.csv")
        self._save_global_vector(gr.R_global, self.maps.assembly_map, self.global_dir / "R_global.csv")
        self._save_global_vector(gr.R_residual, self.maps.assembly_map, self.global_dir / "R_residual.csv")

    def _save_global_matrix(self, mat, dof_map, filepath, *, value_label="K Value", label=None):
        if mat is None or dof_map is None:
            return

        filepath = Path(filepath)
        reader = MapReader(dof_map)
        field = "condensed_dof" if "cond" in filepath.stem.lower() else "global_dof"
        dof_indices = reader.resolve(field=field, ignore_values=[-1])

        if label is None:
            label = "Condensed DOF" if field == "condensed_dof" else "Global DOF"

        if issparse(mat):
            mat = mat.tocoo()
            row_idx = mat.row
            col_idx = mat.col
            data = mat.data
        else:
            arr = np.atleast_2d(mat)
            row_idx, col_idx = np.nonzero(arr)
            data = arr[row_idx, col_idx]

        try:
            row_dofs = dof_indices[row_idx]
            col_dofs = dof_indices[col_idx]
        except IndexError:
            logger.warning(f"⚠️ DOF mapping mismatch in {filepath.name}")
            row_dofs = row_idx
            col_dofs = col_idx

        df = pd.DataFrame({
            f"Row ({label})": row_dofs,
            f"Column ({label})": col_dofs,
            value_label: data
        })
        df.to_csv(filepath, index=False, float_format="%.12e")


    def _save_global_vector(self, vector, dof_map, filepath, *, label=None):
        if vector is None or dof_map is None:
            return

        filepath = Path(filepath)
        vector = np.asarray(vector).ravel()

        reader = MapReader(dof_map)
        field = "condensed_dof" if "cond" in filepath.stem.lower() else "global_dof"
        dof_indices = reader.resolve(field=field, ignore_values=[-1])

        if label is None:
            label = "Condensed DOF" if field == "condensed_dof" else "Global DOF"

        if len(vector) != len(dof_indices):
            logger.warning(
                f"⚠️ Length mismatch saving {filepath.name}: "
                f"{len(vector)} values vs {len(dof_indices)} DOFs"
            )

        df = pd.DataFrame({
            label: dof_indices,
            "Value": vector
        })
        df.to_csv(filepath, index=False, float_format="%.12e")


    # ──────────────────────────────────────────────────────────────
    # ELEMENTAL RESULTS
    # ──────────────────────────────────────────────────────────────

    def _save_element_results(self):
        er = self.results.elemental_results

        self._save_elemental_matrix(er.K_e, self.element_stiffness_dir, value_label="K_e Value")
        self._save_elemental_vector(er.F_e, self.external_force_dir, value_label="F_e Value")
        self._save_elemental_vector(er.U_e, self.deformation_dir, value_label="U_e Value")
        self._save_elemental_vector(er.R_e, self.reaction_force_dir, value_label="R_e Value")
        self._save_elemental_vector(er.R_residual_e, self.residual_reaction_dir, value_label="R_residual_e Value")

    def _save_elemental_vector(self, vector_list, base_dir, *, value_label="Value"):
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        for i, vec in enumerate(vector_list):
            vec = np.asarray(vec).ravel()
            df = pd.DataFrame({
                "Local DOF": np.arange(len(vec)),
                value_label: vec
            })
            filename = base_dir / f"element_{i:04d}.csv"
            df.to_csv(filename, index=False, float_format="%.12e")

    def _save_elemental_matrix(self, matrix_list, base_dir, *, value_label="K_e Value"):
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        for i, mat in enumerate(matrix_list):
            if mat is None:
                continue

            if issparse(mat):
                mat = mat.tocoo()
                row_idx = mat.row
                col_idx = mat.col
                data = mat.data
            else:
                arr = np.atleast_2d(mat)
                row_idx, col_idx = np.nonzero(arr)
                data = arr[row_idx, col_idx]

            df = pd.DataFrame({
                "Row (Local DOF)": row_idx,
                "Column (Local DOF)": col_idx,
                value_label: data
            })
            filename = base_dir / f"element_{i:04d}.csv"
            df.to_csv(filename, index=False, float_format="%.12e")
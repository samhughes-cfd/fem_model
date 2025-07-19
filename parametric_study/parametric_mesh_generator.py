from pathlib import Path
from typing import Dict, List, Union
import numpy as np
import logging
from mesh_generator import MeshGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class ParametricMeshGenerator:
    def __init__(self,
                 base_dir: Union[str, Path],
                 mesh_generator_class,
                 num_nodes: int,
                 growth_factor: float,
                 element_type: str,
                 quadrature_orders: Dict[str, int],
                 material_template: Dict[str, float],
                 section_template: Dict[str, float],
                 point_load_template: List[float]):
        self.base_dir = Path(base_dir)
        self.mesh_generator_class = mesh_generator_class
        self.num_nodes = num_nodes
        self.growth_factor = growth_factor
        self.element_type = element_type
        self.quadrature_orders = quadrature_orders
        self.material_template = material_template
        self.section_template = section_template
        self.point_load_template = point_load_template
        self.study_index = self._get_study_index()

    def _get_study_index(self) -> int:
        base_path = self.base_dir / "parametric_study" / "parametric_meshes"
        base_path.mkdir(parents=True, exist_ok=True)
        existing = sorted(base_path.glob("study_*"))
        if not existing:
            return 0
        return int(existing[-1].name.split("_")[-1]) + 1

    def _generate_study_dir(self) -> Path:
        study_dir = self.base_dir / "parametric_study" / "parametric_meshes" / f"study_{self.study_index:04d}"
        study_dir.mkdir(parents=True, exist_ok=True)
        return study_dir

    def run_study(self,
                  param: str,
                  base_value: float,
                  scale_range: np.ndarray,
                  fixed_inputs: Dict[str, float]):
        study_dir = self._generate_study_dir()
        job_index = 0

        for scale in scale_range:
            updated_material = self.material_template.copy()
            updated_section = self.section_template.copy()
            updated_point_load = [self.point_load_template.copy()]
            beam_length = fixed_inputs["L"]

            if param == "F":
                updated_point_load[0][3] = base_value * scale
            elif param == "E":
                updated_material["E"] = base_value * scale
            elif param == "Iz":
                updated_section["I_z"] = base_value * scale
            elif param == "L":
                beam_length = base_value * scale
            else:
                raise ValueError(f"Unsupported parameter: {param}")

            generator = self.mesh_generator_class(
                job_index=job_index,
                base_dir=study_dir,
                beam_length=beam_length,
                num_nodes=self.num_nodes,
                growth_factor=self.growth_factor,
                element_type=self.element_type,
                quadrature_orders=self.quadrature_orders,
                material_props=updated_material,
                section_props=updated_section,
                point_loads=updated_point_load
            )
            generator.generate_job()
            job_index += 1

        logging.info(f"Parametric study '{param}' complete with {len(scale_range)} jobs saved to {study_dir}")
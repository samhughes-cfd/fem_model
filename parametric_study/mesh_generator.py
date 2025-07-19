from pathlib import Path
from typing import List, Tuple, Dict, Union
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class MeshGenerator:
    def __init__(self,
                 job_index: int,
                 base_dir: Union[str, Path],
                 beam_length: float,
                 num_nodes: int,
                 growth_factor: float,
                 element_type: str,
                 quadrature_orders: Dict[str, int],
                 material_props: Dict[str, float],
                 section_props: Dict[str, float],
                 point_loads: List[List[float]]):
        self.job_index = job_index
        self.base_dir = Path(base_dir)
        self.beam_length = beam_length
        self.num_nodes = num_nodes
        self.growth_factor = growth_factor
        self.element_type = element_type
        self.quadrature_orders = quadrature_orders
        self.material_props = material_props
        self.section_props = section_props
        self.point_loads = point_loads

        self.job_dir = self.base_dir / f"job_{job_index:04d}"
        self.job_dir.mkdir(parents=True, exist_ok=True)

    def generate_node_positions(self) -> np.ndarray:
        if self.growth_factor == 0:
            return np.linspace(0.0, self.beam_length, self.num_nodes)
        else:
            i = np.linspace(0.0, 1.0, self.num_nodes)
            norm = (np.exp(self.growth_factor * i) - 1.0) / (np.exp(self.growth_factor) - 1.0)
            return (1.0 - norm) * self.beam_length

    def generate_elements(self, node_positions: np.ndarray) -> List[Tuple[int, int]]:
        return [(i, i + 1) for i in range(len(node_positions) - 1)]

    def write_grid_file(self, node_positions: np.ndarray):
        path = self.job_dir / "grid.txt"
        with open(path, "w") as f:
            f.write("[Grid]\n[node_id]   [x]         [y]       [z]\n")
            for i, x in enumerate(node_positions):
                f.write(f"{i:<11}{x:<11.6f}{0.0:<9.1f}{0.0:<9.1f}\n")

    def write_element_file(self, elements: List[Tuple[int, int]]):
        path = self.job_dir / "element.txt"
        ftype = self.element_type
        q = self.quadrature_orders
        with open(path, "w") as f:
            f.write("[Element]\n")
            f.write("".join(f"{h:<18}" for h in [
                "[element_id]", "[node1]", "[node2]", "[element_type]",
                "[axial_order]", "[bending_y_order]", "[bending_z_order]",
                "[shear_y_order]", "[shear_z_order]", "[torsion_order]", "[load_order]"
            ]) + "\n")
            for eid, (n1, n2) in enumerate(elements):
                f.write(f"{eid:<18}{n1:<18}{n2:<18}{ftype:<30}"
                        f"{q['axial']:<18}{q['bending_y']:<18}{q['bending_z']:<18}"
                        f"{q['shear_y']:<18}{q['shear_z']:<18}{q['torsion']:<18}{q['load']:<18}\n")

    def write_material_file(self, num_elements: int):
        path = self.job_dir / "material.txt"
        m = self.material_props
        with open(path, "w") as f:
            f.write("[Material]\n[element_id]  [E]         [G]          [nu]   [rho]\n")
            for i in range(num_elements):
                f.write(f"{i:<13d}{m['E']:<11.1e}{m['G']:<12.4e}{m['nu']:<6.2f}{m['rho']:<8.1f}\n")

    def write_section_file(self, num_elements: int):
        path = self.job_dir / "section.txt"
        s = self.section_props
        with open(path, "w") as f:
            f.write("[Section]\n[element_id]  [A]          [I_x]        [I_y]          [I_z]          [J_t]\n")
            for i in range(num_elements):
                f.write(f"{i:<13d}{s['A']:<12.6f}{s['I_x']:<12.1f}{s['I_y']:<14.5e}{s['I_z']:<14.5e}{s['J_t']:<10.5e}\n")

    def write_point_load_file(self):
        path = self.job_dir / "point_load.txt"
        with open(path, "w") as f:
            f.write("[Point load]\n")
            for row in self.point_loads:
                if len(row) != 9:
                    raise ValueError(f"Each point load entry must have 9 values. Found: {row}")
                f.write("".join(f"{val:<12.5e}" for val in row) + "\n")

    def generate_job(self):
        node_positions = self.generate_node_positions()
        elements = self.generate_elements(node_positions)

        self.write_grid_file(node_positions)
        self.write_element_file(elements)
        self.write_material_file(len(elements))
        self.write_section_file(len(elements))
        self.write_point_load_file()

        logging.info(f"Job {self.job_index:04d} generated at {self.job_dir}")
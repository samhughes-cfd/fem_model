"""
Mesh-generator for a cantilever-beam demo model.

Change (2025-07-07)
-------------------
* save_element_file – ORD_W enlarged to 18 so square-bracket labels are
  aligned, equally spaced, and never touch adjacent labels.
"""
import logging
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# User configuration
# --------------------------------------------------------------------------- #
L: float = 2.0            # Beam length [m]
growth_factor: float = 0  # 0 = uniform; >0 = exponential tip clustering
num_uniform_nodes: int = 501
max_num_nodes: int = 501
# --------------------------------------------------------------------------- #


def generate_mesh(growth: float,
                  max_nodes: int,
                  uniform_nodes: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Return node positions and element connectivity."""
    if growth == 0:
        node_positions = np.linspace(0.0, L, uniform_nodes)
    else:
        i = np.linspace(0.0, 1.0, max_nodes)
        norm = (np.exp(growth * i) - 1.0) / (np.exp(growth) - 1.0)
        node_positions = (1.0 - norm) * L           # cluster toward tip

    elements = [(idx, idx + 1) for idx in range(len(node_positions) - 1)]

    logging.info("Mesh generation successful.")
    logging.info("Growth factor: %s", growth)
    logging.info("Total nodes:   %d", len(node_positions))
    logging.info("Total elements:%d", len(elements))
    return node_positions, elements


# --------------------------------------------------------------------------- #
# File writers
# --------------------------------------------------------------------------- #
def save_grid_file(node_positions: np.ndarray, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "grid.txt")

    with open(path, "w") as f:
        f.write("[Grid]\n")
        f.write(f"{'[node_id]':<12}{'[x]':<12}{'[y]':<10}{'[z]':<10}\n")
        for idx, x in enumerate(node_positions):
            f.write(f"{idx:<12d}{x:<12.6f}{0.0:<10.1f}{0.0:<10.1f}\n")

    logging.info("grid.txt written → %s", path)


def save_element_file(elements: List[Tuple[int, int]],
                      save_dir: str,
                      element_type: str = "EulerBernoulliBeamElement3D") -> None:
    """Write element.txt with uniform, parser-friendly column spacing."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "element.txt")

    # Fixed column widths
    ID_W, NODE_W, TYPE_W = 14, 9, 30   # identification columns
    ORD_W = 18                         # ≥ len('[bending_z_order]') + 1

    # Default quadrature orders
    axial_order = bending_y_order = bending_z_order = 3
    shear_y_order = shear_z_order = 0
    torsion_order = 3
    load_order = 2

    with open(path, "w") as f:
        f.write("[Element]\n")
        # ---- header -------------------------------------------------------
        f.write(
            f"{'[element_id]':<{ID_W}}"
            f"{'[node1]':<{NODE_W}}{'[node2]':<{NODE_W}}"
            f"{'[element_type]':<{TYPE_W}}"
            f"{'[axial_order]':<{ORD_W}}"
            f"{'[bending_y_order]':<{ORD_W}}"
            f"{'[bending_z_order]':<{ORD_W}}"
            f"{'[shear_y_order]':<{ORD_W}}"
            f"{'[shear_z_order]':<{ORD_W}}"
            f"{'[torsion_order]':<{ORD_W}}"
            f"{'[load_order]':<{ORD_W}}\n"
        )
        # ---- rows ---------------------------------------------------------
        for idx, (n1, n2) in enumerate(elements):
            f.write(
                f"{idx:<{ID_W}d}"
                f"{n1:<{NODE_W}d}{n2:<{NODE_W}d}"
                f"{element_type:<{TYPE_W}}"
                f"{axial_order:<{ORD_W}d}"
                f"{bending_y_order:<{ORD_W}d}"
                f"{bending_z_order:<{ORD_W}d}"
                f"{shear_y_order:<{ORD_W}d}"
                f"{shear_z_order:<{ORD_W}d}"
                f"{torsion_order:<{ORD_W}d}"
                f"{load_order:<{ORD_W}d}\n"
            )

    logging.info("element.txt written → %s", path)


def save_material_file(elements: List[Tuple[int, int]], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "material.txt")

    E, G, nu, rho = 2.10e11, 8.10e10, 0.3, 7850

    with open(path, "w") as f:
        f.write("[Material]\n")
        f.write(f"{'[element_id]':<14}{'[E]':<12}{'[G]':<13}{'[nu]':<7}{'[rho]':<10}\n")
        for idx in range(len(elements)):
            f.write(f"{idx:<14d}{E:<12.1e}{G:<13.4e}{nu:<7.1f}{rho:<10d}\n")

    logging.info("material.txt written → %s", path)


def save_section_file(elements: List[Tuple[int, int]], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "section.txt")

    A = 1.3075519589902397e-3  # Cross-sectional area [m²]
    I_x = 0.0
    I_y = 3.2340029e-7
    I_z = 2.0876865e-6
    J_t = 2.606727e-8

    # Field widths
    ID_W = 14
    A_W = 14
    IX_W = 14
    IY_W = 18
    IZ_W = 18
    JT_W = 14

    with open(path, "w") as f:
        f.write("[Section]\n")
        f.write(
            f"{'[element_id]':<{ID_W}}"
            f"{'[A]':<{A_W}}"
            f"{'[I_x]':<{IX_W}}"
            f"{'[I_y]':<{IY_W}}"
            f"{'[I_z]':<{IZ_W}}"
            f"{'[J_t]':<{JT_W}}\n"
        )
        for idx in range(len(elements)):
            f.write(
                f"{idx:<{ID_W}d}"
                f"{A:<{A_W}.5f}"
                f"{I_x:<{IX_W}.1f}"
                f"{I_y:<{IY_W}.5e}"
                f"{I_z:<{IZ_W}.5e}"
                f"{J_t:<{JT_W}.5e}\n"
            )

    logging.info("section.txt written → %s", path)

# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        nodes, elems = generate_mesh(growth_factor, max_num_nodes, num_uniform_nodes)
        print("Nodes :", len(nodes))
        print("Elems :", len(elems))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("pre_processing", "mesh_library", "meshes", f"mesh_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        save_grid_file(nodes,      save_dir)
        save_element_file(elems,   save_dir)
        save_material_file(elems,  save_dir)
        save_section_file(elems,   save_dir)

        logging.info("All mesh files saved to ‘%s’.", save_dir)
    except Exception as exc:
        logging.error("Mesh generation failed: %s", exc, exc_info=True)


if __name__ == "__main__":
    main()
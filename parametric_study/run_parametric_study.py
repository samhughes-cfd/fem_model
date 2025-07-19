import numpy as np
from parametric_mesh_generator import ParametricMeshGenerator
from mesh_generator import MeshGenerator
from pathlib import Path

# Base values
F_base = 500           # N
E_base = 2e11          # Pa
Iz_base = 2.08769e-6   # m^4
L_base = 1.0           # m

# Logarithmic scale range
scales = np.logspace(-1, 1, 10)

# Fixed values
fixed_inputs = {
    "F": F_base,
    "E": E_base,
    "Iz": Iz_base,
    "L": L_base
}

material_template = {
    "E": E_base,
    "G": 79.3e9,
    "nu": 0.3,
    "rho": 7850.0
}

section_template = {
    "A": 0.005,
    "I_x": 0.0,
    "I_y": 0.0,
    "I_z": Iz_base,
    "J_t": 1e-6
}

point_load_template = [0, 9, 2, F_base, 1, 1.0, 0.0, 0.0, 0]  # Tip force in y

quadrature_orders = {
    "axial": 2,
    "bending_y": 2,
    "bending_z": 2,
    "shear_y": 2,
    "shear_z": 2,
    "torsion": 2,
    "load": 2
}

pmg = ParametricMeshGenerator(
    base_dir=Path(__file__).parent,
    mesh_generator_class=MeshGenerator,
    num_nodes=10,
    growth_factor=0.0,
    element_type="EBBeam3D2",
    quadrature_orders=quadrature_orders,
    material_template=material_template,
    section_template=section_template,
    point_load_template=point_load_template
)

for param in ["F", "E", "Iz", "L"]:
    pmg.run_study(
        param=param,
        base_value=fixed_inputs[param],
        scale_range=scales,
        fixed_inputs=fixed_inputs
    )

from dataclasses import dataclass
from typing import Any, List

@dataclass
class ElementStiffnessMatrixContainer:
    K_e: Any  # Element stiffness matrix
    xi: List[float]  # Integration point coordinates
    shape_operators: List[Any]
    strain_operators: List[Any]
    material_operators: List[Any]

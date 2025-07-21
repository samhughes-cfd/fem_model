from dataclasses import dataclass
from typing import Any, List

@dataclass
class ElementForceVectorContainer:
    F_e: Any  # Element force vector
    xi: List[float]  # Integration point coordinates
    shape_operators: List[Any]
    strain_operators: List[Any]
    material_operators: List[Any]

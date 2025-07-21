from dataclasses import dataclass
from typing import Any
from .element_stiffness_matrix_container import ElementStiffnessMatrixContainer
from .element_force_vector_container import ElementForceVectorContainer

@dataclass
class ElementContainer:
    element: Any  # The original element object (with ID, nodes, etc.)
    stiffness_data: ElementStiffnessMatrixContainer
    force_data: ElementForceVectorContainer

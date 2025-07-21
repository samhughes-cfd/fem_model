from dataclasses import dataclass
from typing import List
from .element_container import ElementContainer

@dataclass
class MeshContainer:
    element_containers: List[ElementContainer]  # Index = element ID

    def get(self, element_id: int) -> ElementContainer:
        return self.element_containers[element_id]

    def __iter__(self):
        return iter(self.element_containers)

    def __len__(self):
        return len(self.element_containers)

# processing_OOP\static\results\containers\gaussian_results.py

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class GaussianResults:
    gauss_coords: Optional[List[np.ndarray]] = None
    # Each: List of (n_gauss_points, dim) arrays per element

    internal_forces: Optional[List[List[np.ndarray]]] = None
    # Shape: List[element] -> List[gauss_point] -> np.ndarray(shape=(6,))
    # Meaning: [ [ [N, Vy, Vz, T, My, Mz] for gp in element ] for element in mesh ]

    strain: Optional[List[List[np.ndarray]]] = None
    # Shape: List[element] -> List[gauss_point] -> np.ndarray(shape=(n_strain_components,))

    stress: Optional[List[List[np.ndarray]]] = None
    # Shape: List[element] -> List[gauss_point] -> np.ndarray(shape=(n_stress_components,))

    internal_energy_density: Optional[List[List[np.ndarray]]] = None
    # Shape: List[element] -> List[gauss_point] -> np.ndarray(shape=(1,)) or np.ndarray(())
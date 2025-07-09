# processing_OOP\static\results\compute_primary\element_formulation_processor.py

from typing import Sequence, List, Union
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np


class ElementFormulationProcessor:
    """
    Handles raw element-level formulation data and recovers Fe and Ke.

    Responsibilities:
    - Convert Ke from COO to CSR (if necessary)
    - Return cleaned lists of Ke and Fe
    """

    def __init__(
        self,
        K_e: Sequence[Union[csr_matrix, coo_matrix]],
        F_e: Sequence[np.ndarray],
    ):
        self.K_e = list(K_e)
        self.F_e = list(F_e)

    def process(self) -> tuple[List[csr_matrix], List[np.ndarray]]:
        if len(self.K_e) != len(self.F_e):
            raise ValueError("Mismatch between number of stiffness matrices and force vectors.")

        Ke_list = [
            ke.tocsr() if not isinstance(ke, csr_matrix) else ke
            for ke in self.K_e
        ]

        Fe_list = [fe.reshape(-1) if fe.ndim != 1 else fe.copy() for fe in self.F_e]

        for i, (ke, fe) in enumerate(zip(Ke_list, Fe_list)):
            if ke.shape[0] != ke.shape[1]:
                raise ValueError(f"Element {i}: stiffness matrix is not square: {ke.shape}")

        return Ke_list, Fe_list
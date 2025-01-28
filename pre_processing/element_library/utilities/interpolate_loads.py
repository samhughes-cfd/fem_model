# pre_processing\element_library\utilities\interpolate_loads.py

import numpy as np

def interpolate_loads(x_phys: float, loads_array: np.ndarray) -> np.ndarray:
    """
    Interpolates distributed loads from a tabular dataset at a given physical position x_phys.

    Parameters:
    - x_phys (float): The physical x-coordinate where loads need to be interpolated.
    - loads_array (np.ndarray): A 2D array where:
        Column 0 = x-coordinates of loads
        Columns 1-3 = (y, z) coordinates (unused)
        Columns 4-6 = (Fx, Fy, Fz) forces
        Columns 7-9 = (Mx, My, Mz) moments

    Returns:
    - np.ndarray: Interpolated forces and moments as a (6,) array [Fx, Fy, Fz, Mx, My, Mz].
    """
    x_loads = loads_array[:, 0]  # Extract x-coordinates of load points

    # Vectorized interpolation for all 6 components (Fx, Fy, Fz, Mx, My, Mz)
    interpolated_forces = np.interp(x_phys, x_loads, loads_array[:, 3:9], left=0, right=0)

    return interpolated_forces  # Shape: (6,)

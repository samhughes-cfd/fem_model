import numpy as np

def interpolate_loads(x_phys: np.ndarray, loads_array: np.ndarray) -> np.ndarray:
    """
    Fully vectorized interpolation of distributed loads for multiple physical positions (x_phys).

    Parameters:
    - x_phys (np.ndarray or float): The physical x-coordinates where loads need to be interpolated.
    - loads_array (np.ndarray): A 2D array where:
        Column 0 = x-coordinates of loads
        Columns 1-3 = (y, z) coordinates (unused)
        Columns 3-9 = (Fx, Fy, Fz, Mx, My, Mz) forces & moments

    Returns:
    - np.ndarray: Interpolated forces and moments as (6,) array for a scalar input or (N, 6) array for multiple x_phys.
    """

    x_loads = loads_array[:, 0]  # Extract x-coordinates of load points
    force_components = loads_array[:, 3:9]  # Extract all six force/moment components

    # Ensure x_phys is always a 1D array for vectorization
    x_phys = np.atleast_1d(x_phys)  # Converts scalars into arrays

    # Perform vectorized interpolation for all six force components at once
    interpolated_forces = np.vstack([
        np.interp(x_phys, x_loads, force_components[:, i], left=0, right=0) # NumPy linear interpolation
        for i in range(6)  # Iterate over force/moment components
    ]).T  # Transpose to get shape (N, 6)

    # Return (6,) for scalar input or (N, 6) for multiple x_phys
    return interpolated_forces if x_phys.shape[0] > 1 else interpolated_forces.squeeze()
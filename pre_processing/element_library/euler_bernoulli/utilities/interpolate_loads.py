# pre_processing\element_library\euler_bernoulli\utilities\interpolate_loads.py

import numpy as np
from scipy import interpolate
from dataclasses import dataclass, field
from typing import Tuple

@dataclass(frozen=True)
class LoadInterpolationOperator:
    """
    Interpolates distributed loads for 6-DOF finite elements with:
    - Boundary-aware extrapolation control
    - Automatic monotonic sorting of position data
    - Adaptive fallback of interpolation scheme based on Gauss point resolution

    Parameters
    ----------
    distributed_loads_array : ndarray of shape (N, 9)
        Array of distributed loads. Columns:
        [x, y, z, Fx, Fy, Fz, Mx, My, Mz].

    boundary_mode : {'error', 'clamp', 'zero'}, default='error'
        Extrapolation behavior for queries outside x-range of provided data:
        - 'error' : raise an exception.
        - 'clamp' : use end values at boundaries.
        - 'zero'  : return zero beyond data bounds.

    interpolation_order : {'nearest', 'linear', 'quadratic', 'cubic'}, default='linear'
        Requested interpolation scheme. If the order requires more data points
        than available or exceeds the integration accuracy of the element's
        quadrature rule, it will be downgraded automatically.

    n_gauss_points : int, default=3
        Number of Gauss points used in the element’s quadrature rule.
        This restricts the maximum interpolated polynomial degree to
        ensure exact integration.

    Attributes
    ----------
    _active_components : tuple of bool
        Boolean mask indicating which load/moment components are non-zero.

    _interpolators : dict of int -> callable
        Dictionary of interpolation functions for each active component (0–5).

    Raises
    ------
    ValueError
        If inputs are invalid, or fallback to a safe interpolation scheme fails.
    """
    distributed_loads_array: np.ndarray
    boundary_mode: str = "error"
    interpolation_order: str = "linear"
    n_gauss_points: int = 3

    _active_components: Tuple[bool] = field(init=False)
    _interpolators: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self._validate_loads()
        self._validate_interpolation_order()
        self._ensure_monotonic_positions()
        self._identify_active_components()
        self._build_interpolators()

    def _validate_loads(self) -> None:
        """Ensure load array is numeric, finite, and shaped [N, 9]."""
        if self.distributed_loads_array.ndim != 2 or self.distributed_loads_array.shape[1] != 9:
            raise ValueError(f"Expected shape (N, 9), got {self.distributed_loads_array.shape}.")
        if not np.isfinite(self.distributed_loads_array).all():
            raise ValueError("Load array contains NaN or Inf.")
        if np.any(np.diff(self.distributed_loads_array[:, 0]) == 0):
            raise ValueError("Duplicate x-coordinates detected in load array.")

    def _validate_interpolation_order(self) -> None:
        """
        Downgrade interpolation order based on available data and Gauss point resolution.

        Ensures:
        - Enough data points exist for the selected scheme.
        - Interpolated polynomial is exactly integrable by the selected quadrature.
        """
        supported_kinds = ['nearest', 'linear', 'quadratic', 'cubic']
        min_points_required = {'nearest': 1, 'linear': 2, 'quadratic': 3, 'cubic': 4}
        degree = {'nearest': 0, 'linear': 1, 'quadratic': 2, 'cubic': 3}
        max_integrable_degree = 2 * self.n_gauss_points - 1
        n_points = self.distributed_loads_array.shape[0]

        valid_orders = [
            k for k in supported_kinds
            if min_points_required[k] <= n_points and degree[k] <= max_integrable_degree
        ]

        if not valid_orders:
            raise ValueError(f"No valid interpolation scheme for {n_points} points and "
                             f"{self.n_gauss_points} Gauss points.")

        if self.interpolation_order in valid_orders:
            return  # Valid request
        fallback_order = valid_orders[-1]
        object.__setattr__(self, 'interpolation_order', fallback_order)

    def _ensure_monotonic_positions(self) -> None:
        """
        Ensure strictly increasing positions in x, y, z columns.
        Sorting is performed along the first non-monotonic axis.
        """
        for dim, axis in enumerate(['x', 'y', 'z']):
            coords = self.distributed_loads_array[:, dim]
            if not np.all(np.diff(coords) > 0):
                sort_idx = np.argsort(coords)
                sorted_array = self.distributed_loads_array[sort_idx]
                object.__setattr__(self, 'distributed_loads_array', sorted_array)
                if not np.all(np.diff(sorted_array[:, dim]) > 0):
                    raise ValueError(f"Non-monotonic positions persist along {axis}-axis after sorting.")

    def _identify_active_components(self) -> None:
        """Mark which of Fx, Fy, Fz, Mx, My, Mz are non-zero in the data."""
        active = np.any(self.distributed_loads_array[:, 3:9] != 0.0, axis=0)
        object.__setattr__(self, '_active_components', tuple(active))

    def _build_interpolators(self) -> None:
        """
        Create interpolation functions for each active component.
        Uses SciPy's interp1d with the specified interpolation order.
        """
        x = self.distributed_loads_array[:, 0]
        for i, is_active in enumerate(self._active_components):
            if not is_active:
                continue
            y = self.distributed_loads_array[:, 3 + i]
            fill_val = self._get_fill_values(y)
            interp_fn = interpolate.interp1d(
                x,
                y,
                kind=self.interpolation_order,
                bounds_error=(self.boundary_mode == "error"),
                fill_value=fill_val,
                assume_sorted=True
            )
            self._interpolators[i] = interp_fn

    def _get_fill_values(self, y: np.ndarray):
        """Define fill values based on boundary mode."""
        if self.boundary_mode == "clamp":
            return y[0], y[-1]
        elif self.boundary_mode == "zero":
            return 0.0
        return None  # For 'error'

    def interpolate(self, x_phys: np.ndarray) -> np.ndarray:
        """
        Interpolate distributed load components at query positions.

        Parameters
        ----------
        x_phys : ndarray of shape (M,)
            Positions in physical space at which to interpolate.

        Returns
        -------
        ndarray of shape (M, 6)
            Interpolated [Fx, Fy, Fz, Mx, My, Mz] values at each query point.

        Raises
        ------
        ValueError
            If any query point lies outside bounds and boundary_mode == 'error'.
        """
        x_phys = np.atleast_1d(np.asarray(x_phys, dtype=np.float64))
        result = np.zeros((x_phys.shape[0], 6), dtype=np.float64)

        for i, is_active in enumerate(self._active_components):
            if is_active:
                try:
                    result[:, i] = self._interpolators[i](x_phys)
                except ValueError as e:
                    x_min = self.distributed_loads_array[0, 0]
                    x_max = self.distributed_loads_array[-1, 0]
                    raise ValueError(
                        f"Interpolation failed for component index {i} at positions {x_phys}. "
                        f"Valid range: [{x_min:.3e}, {x_max:.3e}]. "
                        f"Details: {e}"
                    ) from e

        return result.squeeze() if result.shape[0] == 1 else result
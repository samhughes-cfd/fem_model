# post_processing\validation_visualisers\deflection_tables\roarks_formulas._distriuted.py

import warnings
import numpy as np
from scipy.integrate import quad

def roark_load_intensity(x, L, w, load_type):
    """
    Returns q(x) for the chosen load type (UDL, triangular, parabolic),
    with maximum at x=0 (the fixed end) and zero at x=L for non-UDL.
    """
    if load_type == "udl":
        # Uniform load
        return w * np.ones_like(x)
    elif load_type == "triangular":
        # q(x) = w(1 - x/L)
        return w * (1 - x/L)
    elif load_type == "parabolic":
        # q(x) = w(1 - x/L)^2
        return w * (1 - x/L)**2
    else:
        raise ValueError("Invalid load_type. Must be 'udl', 'triangular', or 'parabolic'.")

def roark_shear(x, L, w, load_type):
    """
    Roark's-style shear force V(x) for a cantilever:
      V(x) = - int_{t=x..L} q(t) dt.
    Returns array of V(x) for each x in input array.

    Uses closed-form integrals for UDL, triangular, parabolic.
    """
    if load_type == "udl":
        # V(x) = - w (L - x)
        return -w*(L - x)
    elif load_type == "triangular":
        # V(x) = - w [ L/2 - x + x^2/(2L)]
        return - w*( (L/2) - x + x**2/(2*L) )
    elif load_type == "parabolic":
        # V(x) = - w [ (L/3) - x + x^2/L - x^3/(3L^2 ) ]
        return - w*( (L/3) - x + x**2/L - x**3/(3*L**2) )
    else:
        raise ValueError("Invalid load_type for shear")

def roark_moment(x, L, w, load_type):
    """
    Roark's-style bending moment M(x) for a cantilever:
      M'(x) = - V(x),  with M(L)=0.
    """
    if load_type == "udl":
        # M(x) = - w ( Lx - x^2/2 )
        return - w*( L*x - x**2/2 )
    elif load_type == "triangular":
        # M(x) = - w [ L^2/6 - (xL)/2 + x^2/2 - x^3/(6L) ]
        return - w * ( (L**2)/6 - (x*L)/2 + x**2/2 - (x**3)/(6*L) )
    elif load_type == "parabolic":
        # M(x) = - w [ L^2/12 - (xL)/3 + x^2/2 - x^3/(3L) + x^4/(12L^2) ]
        return - w * ( (L**2)/12 - (x*L)/3 + (x**2)/2 - (x**3)/(3*L) + (x**4)/(12*L**2) )
    else:
        raise ValueError("Invalid load_type for moment")

def roark_rotation(x_array, L, w, E, I, load_type):
    """
    Rotation: theta_z(x) = (1/(E I)) * ∫[0..x] M(t) dt, with θ(0)=0.
    We'll do numeric integration for all load types for consistency.
    """
    def M_over_EI(t):
        # Evaluate moment at t and divide by (E*I)
        val = roark_moment(np.array([t]), L, w, load_type)[0]
        return val/(E*I)

    thetas = []
    for xval in x_array:
        val, _ = quad(M_over_EI, 0, xval, limit=100)
        thetas.append(val)
    return np.array(thetas)

def roark_deflection(x_array, L, w, E, I, load_type):
    """
    Computes deflection: u_y(x) = ∫_0^x θ_z(s) ds.
    Handles integration errors by adjusting numerical tolerances.
    """
    rotation_values = roark_rotation(x_array, L, w, E, I, load_type)

    def rotation_func(s):
        return np.interp(s, x_array, rotation_values)

    u_vals = []
    for xval in x_array:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UserWarning)  # Convert warnings to exceptions
            try:
                val, _ = quad(rotation_func, 0, xval, limit=100, epsabs=1e-6, epsrel=1e-6)
            except UserWarning as e:
                print(f"⚠️ Warning: Integration failed at x = {xval:.3f} ({e}), using fallback method")
                idx = np.searchsorted(x_array, xval)
                val = np.trapz(rotation_values[:idx], x_array[:idx])  # Use Trapezoidal Rule

        u_vals.append(val)

    return np.array(u_vals)

    return np.array(u_vals)

def roark_distributed_load_response(x, L, E, I, w, load_type):
    """
    Returns a dictionary of distributed-load responses:
       "intensity":  q(x),
       "shear":      V(x),
       "moment":     M(x),
       "rotation":   θ(x),
       "deflection": u_y(x)
    for 'udl', 'triangular', or 'parabolic'.
    """
    if load_type not in ("udl", "triangular", "parabolic"):
        raise ValueError("Invalid load_type: must be 'udl','triangular','parabolic'.")

    q_vals      = roark_load_intensity(x, L, w, load_type)
    shear_vals  = roark_shear(x, L, w, load_type)
    moment_vals = roark_moment(x, L, w, load_type)
    rot_vals    = roark_rotation(x, L, w, E, I, load_type)
    defl_vals   = roark_deflection(x, L, w, E, I, load_type)

    return {
        "intensity":  q_vals,
        "shear":      shear_vals,
        "moment":     moment_vals,
        "rotation":   rot_vals,
        "deflection": defl_vals
    }
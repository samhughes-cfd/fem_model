# post_processing\validation_visualisers\deflection_tables\roarks_formulas_point.py

import numpy as np

def roark_point_load_intensity(x, L, P, a):
    """
    Defines the point load intensity q(x), which is zero everywhere except at x=a.
    The function approximates the Dirac delta function by setting q(x) = P at the
    closest discrete point to x=a.
    """
    q = np.zeros_like(x)
    idx_a = np.argmin(abs(x - a))  # Find index closest to x=a
    q[idx_a] = -P  # Negative because load acts downward
    return q

def roark_point_load_shear(x, L, P, a):
    """
    Roark's shear V(x) for a cantilever of length L,
    with a point load P at x=a.
    Returns array of shear values for each x in input array.
    """
    V = np.zeros_like(x)
    for i, xv in enumerate(x):
        if xv < a:
            V[i] = -P
        else:
            V[i] = 0
    return V

def roark_point_load_moment(x, L, P, a):
    """
    Roark's bending moment M(x) for a cantilever of length L,
    with a point load P at x=a.
    Piecewise:
        M(x) = -P(a - x), for 0 <= x < a
        M(x) = 0         for a <= x <= L
    """
    M = np.zeros_like(x)
    for i, xv in enumerate(x):
        if xv < a:
            M[i] = -P * (a - xv)
        else:
            M[i] = 0
    return M

def roark_point_load_rotation(x, L, E, I, P, a):
    """
    Roark's slope (rotation) theta_z(x) for a cantilever with a point load at x=a.
    Piecewise:
      for 0 <= x < a:
         theta_z(x) = - (P * x)/(2 E I) [2a - x]
      for a <= x <= L:
         constant = theta_z(a)
    """
    theta = np.zeros_like(x)

    def slope_region1(xx):
        return -(P * xx)/(2.0 * E * I) * (2.0*a - xx)

    # Evaluate slope at x=a => constant for x>=a
    theta_a = slope_region1(a)

    for i, xv in enumerate(x):
        if xv < a:
            theta[i] = slope_region1(xv)
        else:
            theta[i] = theta_a
    return theta

def roark_point_load_deflection(x, L, E, I, P, a):
    """
    Roark's deflection u_y(x) for a cantilever with a point load P at x=a.
    Piecewise:
      for 0 <= x < a:
         u_y(x) = - (P x^2)/(6 E I) [3a - x]
      for a <= x <= L:
         linear extension from x=a =>  u_y(a) + theta(a)*(x - a)
    """
    u = np.zeros_like(x)

    def defl_region1(xx):
        return -(P * xx**2)/(6.0 * E * I) * (3.0*a - xx)

    u_a = defl_region1(a)

    def slope_region1(xx):
        return -(P * xx)/(2.0 * E * I) * (2.0*a - xx)
    theta_a = slope_region1(a)

    for i, xv in enumerate(x):
        if xv < a:
            u[i] = defl_region1(xv)
        else:
            u[i] = u_a + theta_a*(xv - a)
    return u

def roark_point_load_response(x, L, E, I, P, load_type):
    """
    Returns a dictionary of:
      {
        "intensity":  q(x),
        "shear":      V(x),
        "moment":     M(x),
        "rotation":   theta_z(x),
        "deflection": u_y(x)
      }
    for a single concentrated load at either:
      load_type='end'     => a = L
      load_type='mid'     => a = L/2
      load_type='quarter' => a = L/4
    """
    if load_type not in ("end","mid","quarter"):
        raise ValueError("Invalid load_type, must be 'end','mid','quarter'.")

    # Determine the load location a
    if load_type == "end":
        a = L
    elif load_type == "mid":
        a = L / 2
    else:  # "quarter"
        a = L / 4

    # Compute piecewise arrays
    qvals = roark_point_load_intensity(x, L, P, a)
    Vvals = roark_point_load_shear(x, L, P, a)
    Mvals = roark_point_load_moment(x, L, P, a)
    thetavals = roark_point_load_rotation(x, L, E, I, P, a)
    uvals = roark_point_load_deflection(x, L, E, I, P, a)

    return {
        "intensity":  qvals,
        "shear":      Vvals,
        "moment":     Mvals,
        "rotation":   thetavals,
        "deflection": uvals,
    }
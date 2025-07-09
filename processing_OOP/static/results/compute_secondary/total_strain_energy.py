# processing_OOP\static\results\compute_secondary\total_strain_energy.py

# ELEMENT RESOLUTION

from typing import Dict

class ComputeTotalStrainEnergyPerElement:
    """
    Computes the total strain energy stored in each finite element by integrating
    the strain energy density over the element domain using numerical quadrature.

    The total strain energy for an element is defined as:

        U_e = ∫_Ω ½ εᵀ(x) σ(x) dΩ = ∫_Ω ½ εᵀ(x) D ε(x) dΩ

    where ε(x) is the strain vector, σ(x) the stress vector, and D the material
    constitutive matrix. In practice, this integral is approximated via Gauss
    quadrature:

        U_e ≈ ∑_g w_g ⋅ w(x_g) ⋅ |J(x_g)|

    where w_g are quadrature weights, w(x_g) is the strain energy density at
    Gauss point g, and |J(x_g)| is the Jacobian determinant (volume scaling).

    This scalar quantity characterizes the internal elastic work stored in each
    element and is used for energy tracking, verification, and error estimation.

    Returns
    -------
    Dict[int, float]
        Element-wise total strain energy (units: J)
    """
    def __init__(self, element_dictionary, gaussian_results):
            self.element_dictionary = element_dictionary
            self.gaussian_results = gaussian_results

    def run(self) -> Dict[int, float]:
        """Returns a mapping of element_id → total strain energy."""
        energy_per_element = {}

        for e_id, elem_data in self.element_dictionary.items():
            xi_gauss, weights = elem_data["integration_points"]
            densities = self.gaussian_results.internal_energy_density[e_id]  # List of scalars

            assert len(xi_gauss) == len(densities)

            # ∫ w(x) dx ≈ ∑ w_i * ρ_i * J
            J = elem_data.get("jacobian", 1.0)  # or compute from geometry if needed
            total_energy = sum(wi * ρ * J for wi, ρ in zip(weights, densities))
            energy_per_element[e_id] = total_energy

        return energy_per_element
# mesher.py

import logging
from pre_processing.element_library.euler_bernoulli.euler_bernoulli_3DOF import EulerBernoulliBeamElement3DOF
# Import or define other element classes as needed

logger = logging.getLogger(__name__)

def build_all_elements(mesh_dictionary, geometry_array, material_array, load_array):
    """
    Instantiate all element objects (e.g. EulerBernoulliBeamElement3DOF) in one place.
    Returns a list of element objects.
    """
    element_ids = mesh_dictionary["element_ids"]
    element_types = mesh_dictionary["element_types"]

    elements = []
    for elem_id, elem_type in zip(element_ids, element_types):
        if elem_type == "EulerBernoulliBeamElement3DOF":
            e = EulerBernoulliBeamElement3DOF(
                element_id=elem_id,
                material_array=material_array,
                geometry_array=geometry_array,
                mesh_dictionary=mesh_dictionary,
                load_array=load_array
            )
            elements.append(e)
        else:
            logger.warning(f"Element type '{elem_type}' not recognized; returning None.")
            elements.append(None)

    return elements

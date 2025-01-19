import numpy as np
import logging
import os
from datetime import datetime

# Beam length
L = 8.0

# Given x-coordinates normalized by 0.8 and scaled by beam length L
x_coords = (np.array([
    0.100084651, 0.100423171, 0.101099885, 0.102114136, 0.103464945, 0.105151005, 0.107170683, 0.109522027,
    0.112202761, 0.115210292, 0.11854171, 0.122193793, 0.126163007, 0.130445513, 0.135037167, 0.139933527,
    0.145129857, 0.150621128, 0.15640203, 0.162466968, 0.168810076, 0.175425217, 0.182305992, 0.189445743,
    0.196837564, 0.204474303, 0.212348572, 0.220452753, 0.228779007, 0.237319278, 0.246065304, 0.255008623,
    0.264140584, 0.273452351, 0.282934918, 0.292579109, 0.302375594, 0.312314897, 0.322387401, 0.332583362,
    0.342892916, 0.353306089, 0.363812807, 0.374402906, 0.38506614, 0.395792193, 0.406570689, 0.4173912,
    0.428243258, 0.439116365, 0.45, 0.460883635, 0.471756742, 0.4826088, 0.493429311, 0.504207807, 0.51493386,
    0.525597094, 0.536187193, 0.546693911, 0.557107084, 0.567416638, 0.577612599, 0.587685103, 0.597624406,
    0.607420891, 0.617065082, 0.626547649, 0.635859416, 0.644991377, 0.653934696, 0.662680722, 0.671220993,
    0.679547247, 0.687651428, 0.695525697, 0.703162436, 0.710554257, 0.717694008, 0.724574783, 0.731189924,
    0.737533032, 0.74359797, 0.749378872, 0.754870143, 0.760066473, 0.764962833, 0.769554487, 0.773836993,
    0.777806207, 0.78145829, 0.784789708, 0.787797239, 0.790477973, 0.792829317, 0.794848995, 0.796535055,
    0.797885864, 0.798900115, 0.799576829, 0.799915349
]) / 0.8) * L

def generate_mesh():
    """
    Generate mesh using given x-coordinates.
    """
    node_positions = x_coords
    elements = [(idx + 1, idx + 2) for idx in range(len(node_positions) - 1)]
    return node_positions, elements

def save_mesh_to_file(node_positions, elements, save_directory='mesh_library/meshes'):
    """
    Save the mesh nodes and elements to a text file.
    """
    os.makedirs(save_directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mesh_{timestamp}.txt"
    filepath = os.path.join(save_directory, filename)
    spacing_adjustment = 1

    with open(filepath, 'w') as f:
        f.write("[element_types]\n")
        f.write("EulerBernoulliBeamElement\n\n")
        f.write("[node_ids]   [x]          [y]        [z]        [connectivity]\n")

        for idx, x_pos in enumerate(node_positions):
            node_id = idx + 1
            connectivity = "-" if idx == len(node_positions) - 1 else f"({node_id}, {node_id + 1})"
            f.write(f"{node_id:<{12+spacing_adjustment}}{x_pos:<{12+spacing_adjustment}.6f}{0.0:<{10+spacing_adjustment}.1f}{0.0:<{10+spacing_adjustment}.1f}{connectivity}\n")

    logging.info(f"Mesh file saved to '{filepath}'.")

def main():
    """
    Main function to generate and save the mesh.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        nodes, elems = generate_mesh()
        print(f"Number of nodes: {len(nodes)}")
        print(f"Number of elements: {len(elems)}")
        save_mesh_to_file(nodes, elems)
    except Exception as e:
        logging.error(f"An error occurred during mesh generation: {e}")

if __name__ == "__main__":
    main()
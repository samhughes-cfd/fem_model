import numpy as np
import os
import logging
from processing.assembly import assemble_global_system
from processing.boundary_conditions import apply_boundary_conditions
from processing.solver_registry import get_solver_registry

logger = logging.getLogger(__name__)

def run_static_fem_solver(settings, job_name, results_dir):
    """
    Executes a static FEM simulation and saves results.

    Args:
        settings (dict): Simulation settings (mesh, elements, loads, boundary conditions, solver).
        job_name (str): Name of the job directory for storing results.
        results_dir (str): Base directory for storing results.

    Returns:
        dict: Contains primary FEM results including:
            - "element_stiffness_matrices": Element stiffness matrices
            - "nodal_displacements": Computed nodal displacements
            - "nodal_forces": Global force vector
            - "reaction_forces": Reaction forces at constrained DOFs
    """
    primary_results_dir = os.path.join(results_dir, job_name, "primary_results")
    os.makedirs(primary_results_dir, exist_ok=True)

    logger.info(f"Running static FEM analysis for {job_name}.")

    # ✅ Compute total degrees of freedom (always 6 DOFs per node)
    total_dof = len(settings["node_positions"]) * 6  

    # ✅ Assemble global stiffness matrix and force vector
    logger.info("Assembling global stiffness matrix and force vector.")
    K_global, F_global, element_stiffness_matrices = assemble_global_system(settings["elements"], total_dof)

    # ✅ Apply nodal loads
    if "nodal_loads" in settings:
        F_global += settings["nodal_loads"].flatten()  # Ensuring 1D format for vector summation

    # ✅ Apply boundary conditions (fixing incorrect key)
    constrained_dofs = np.where(settings["boundary_conditions"].flatten() == 1)[0]  # Extract 1D constrained DOFs
    K_mod, F_mod = apply_boundary_conditions(K_global, F_global, constrained_dofs)

    # ✅ Solve for nodal displacements
    logger.info(f"Solving system using {settings['solver_name']}.")
    displacements = solve_fem_system(K_mod, F_mod, settings["solver_name"])

    # ✅ Compute reaction forces
    reaction_forces = compute_reaction_forces(K_global, displacements, constrained_dofs)

    # ✅ Save results
    results = {
        "element_stiffness_matrices": element_stiffness_matrices,
        "nodal_displacements": displacements,
        "nodal_forces": F_global,
        "reaction_forces": reaction_forces,
    }
    save_results(results, settings, primary_results_dir)

    return results

def solve_fem_system(K_mod, F_mod, solver_name):
    """Solves the FEM system for nodal displacements using the selected solver."""
    logger.info(f"Solving FEM system using {solver_name}.")

    solver_registry = get_solver_registry()
    if solver_name not in solver_registry:
        raise ValueError(f"Solver '{solver_name}' is not available in the registry.")

    solver_func = solver_registry[solver_name]

    try:
        return solver_func(K_mod, F_mod)
    except Exception as e:
        raise RuntimeError(f"Solver '{solver_name}' encountered an error: {str(e)}")

def compute_reaction_forces(K_global, displacements, constrained_dofs):
    """Computes reaction forces at constrained DOFs."""
    reaction_forces = K_global @ displacements
    return reaction_forces[constrained_dofs]

def save_results(results, settings, results_dir):
    """Saves primary results as text files with metadata."""
    logger.info("Saving primary results.")

    metadata = [
        f"Simulation Type: Static",
        f"Solver Used: {settings['solver_name']}",
        f"Element Types: {', '.join(settings['element_types'])}",
        f"Number of Nodes: {len(settings['node_positions'])}",
        f"Number of Elements: {len(settings['elements'])}",
    ]

    _save_array("nodal_displacements.txt", metadata, results["nodal_displacements"], results_dir)
    _save_array("nodal_forces.txt", metadata, results["nodal_forces"], results_dir)
    _save_array("reaction_forces.txt", metadata, results["reaction_forces"], results_dir)

    for element_id, Ke in results["element_stiffness_matrices"].items():
        _save_array(f"element_stiffness_{element_id}.txt", metadata, Ke, results_dir)

def _save_array(filename, metadata, array, results_dir):
    """Helper function to save arrays as text files with metadata."""
    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w") as f:
        for line in metadata:
            f.write(f"# {line}\n")
        np.savetxt(f, array, delimiter=",", fmt="%.6f")

    logger.info(f"Saved: {filepath}")
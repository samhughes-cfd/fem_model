import os
import numpy as np
from fem_io import read_stiffness_matrices, read_force_vectors, save_matrix_to_file, save_vector_to_file
from fem_assembly import get_total_nodes, assemble_global_system, apply_boundary_conditions
from fem_static_condensation import static_condensation, reconstruct_full_solution
from fem_solver import solve_sparse_system
from fem_logging import ensure_output_directory, log_system_diagnostics

if __name__ == "__main__":
    # Determine the directory of the current script.
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define input and output directories.
    input_dir = os.path.join(script_dir, "input")
    output_dir = os.path.join(script_dir, "output")
    ensure_output_directory(output_dir)

    # Build absolute paths for input files.
    stiffness_file = os.path.join(input_dir, "stiffness_matrices.txt")
    force_file = os.path.join(input_dir, "force_vectors.txt")

    # Read element data.
    stiffness_matrices = read_stiffness_matrices(stiffness_file)
    force_vectors = read_force_vectors(force_file)

    # Assemble the global system.
    global_K, global_F = assemble_global_system(stiffness_matrices, force_vectors, output_dir)
    save_matrix_to_file(global_K, "global_K.txt", output_dir)
    save_vector_to_file(global_F, "global_F.txt", output_dir)

    # Determine number of nodes and fix all DOFs of Node 0.
    num_nodes = get_total_nodes(stiffness_matrices)
    fixed_dofs = list(range(6))

    # Apply boundary conditions.
    K_mod, F_mod = apply_boundary_conditions(global_K, global_F, fixed_dofs, output_dir)
    save_matrix_to_file(K_mod, "K_mod.txt", output_dir)
    save_vector_to_file(F_mod, "F_mod.txt", output_dir)

    # Log diagnostics for the modified system.
    log_system_diagnostics(K_mod, F_mod, fixed_dofs, output_dir, label="Modified System (K_mod, F_mod)")

    # Static condensation step.
    active_dofs, inactive_dofs, K_condensed, F_condensed = static_condensation(K_mod, F_mod, tol=1e-12)
    save_matrix_to_file(K_condensed, "K_condensed.txt", output_dir)
    save_vector_to_file(F_condensed, "F_condensed.txt", output_dir)

    # Log diagnostics for the condensed system.
    log_system_diagnostics(K_condensed, F_condensed, fixed_dofs, output_dir, label="Condensed System (K_condensed, F_condensed)")

    # Solve for displacements using the sparse solver on the condensed system.
    try:
        d_reduced = solve_sparse_system(K_condensed, F_condensed, output_dir, use_iterative=False)
    except Exception as e:
        print(f"⚠️ Solver error: {e}")
        exit(1)

    # Reconstruct the full displacement vector using the mapping.
    full_displacements = reconstruct_full_solution(active_dofs, d_reduced, global_K.shape[0])
    
    # Compute reaction forces on the full system.
    reaction_forces = np.dot(global_K, full_displacements) - global_F
    save_vector_to_file(full_displacements, "global_U.txt", output_dir)
    save_vector_to_file(reaction_forces, "global_R.txt", output_dir)

    # Final system is K_mod, F_mod, global_U

    print("✅ Solver completed. Results saved in the 'output' directory.")
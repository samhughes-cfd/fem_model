import numpy as np
import os
from fem_logging import log_system_diagnostics

def get_total_nodes(elements_stiffness):
    # Assuming sequential node numbering: (number of elements + 1)
    return len(elements_stiffness) + 1

def assemble_global_system(elements_stiffness, elements_force, output_dir):
    num_nodes = get_total_nodes(elements_stiffness)
    global_size = num_nodes * 6
    global_K = np.zeros((global_size, global_size))
    global_F = np.zeros(global_size)

    dof_mapping_log_path = os.path.join(output_dir, "dof_mapping_log.txt")
    with open(dof_mapping_log_path, 'w', encoding="utf-8") as log_file:
        log_file.write("# Local to Global DOF Mapping Log\n\n")
        for element_id, K in elements_stiffness.items():
            # For each element, two nodes are involved: node i and node i+1.
            node1 = element_id
            node2 = element_id + 1

            # Global DOF breakdown for node1:
            node1_global_start = node1 * 6
            node1_global_end = node1_global_start + 5  # inclusive

            # Global DOF breakdown for node2:
            node2_global_start = node2 * 6
            node2_global_end = node2_global_start + 5  # inclusive

            # Combined global DOF range for the element (12 DOFs total):
            combined_global_start = node1_global_start
            combined_global_end = node2_global_end  # inclusive
            # For array slicing, we'll use combined_global_end+1

            # Update the global stiffness matrix:
            global_K[combined_global_start:combined_global_end+1,
                     combined_global_start:combined_global_end+1] += K

            # Write a detailed mapping log:
            log_file.write(f"Element ID: {element_id}\n")
            log_file.write(f"  Local Node 0 (Local DOFs 0-5) -> Global Node {node1} (Global DOFs {node1_global_start}-{node1_global_end})\n")
            log_file.write(f"  Local Node 1 (Local DOFs 6-11) -> Global Node {node2} (Global DOFs {node2_global_start}-{node2_global_end})\n")
            log_file.write(f"  Combined: Local DOFs (0-11) -> Global DOFs {combined_global_start}-{combined_global_end}\n\n")
        
        # Assemble the global force vector:
        for element_id, F in elements_force.items():
            start_dof = element_id * 6
            end_dof = start_dof + 12
            global_F[start_dof:end_dof] += F

    # Zero out numerical noise (robustness against near-zero terms)
    global_K[np.abs(global_K) < 1e-14] = 0.0

    return global_K, global_F

def apply_boundary_conditions(K, F, fixed_dofs, output_dir):
    K_mod = np.delete(np.delete(K, fixed_dofs, axis=0), fixed_dofs, axis=1)
    F_mod = np.delete(F, fixed_dofs)

    log_system_diagnostics(K_mod, F_mod, fixed_dofs, output_dir)
    return K_mod, F_mod
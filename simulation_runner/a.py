    def compute_primary_results(self, K_global, F_global, K_mod, F_mod, 
                                K_cond, F_cond, U_cond, U_global):
        """
        Computes primary results including reaction forces, displacements, nodal rotations (θ_z),
        and bending moments (M_z) for Euler-Bernoulli beam elements.
        
        Extensive logging is performed to compare the new post-processed global results with the original
        (pre-processed) results.
        
        Returns:
            Tuple: (global_results, element_results)
        """
        logger.info("Computing primary results (extensive logging enabled)...")

        # Retrieve boundary DOFs (for the full system) without modifying matrices.
        _, _, bc_dofs = apply_boundary_conditions(K_global, F_global)

        # -----------------------------------------------------
        # ORIGINAL GLOBAL REACTION FORCES
        # -----------------------------------------------------
        R_global = np.zeros_like(F_global)
        R_global[bc_dofs] = -F_global[bc_dofs]
        logger.debug("Original R_global computed. Shape: %s, Norm: %.3e", R_global.shape, np.linalg.norm(R_global))

        # Compute R_cond using the condensed system equation.
        R_cond = K_cond @ U_cond - F_cond
        logger.debug("Original R_cond computed from K_cond @ U_cond - F_cond. Shape: %s, Norm: %.3e", 
                     R_cond.shape, np.linalg.norm(R_cond))

        # -----------------------------------------------------
        # STORE INITIAL GLOBAL RESULTS (Pre-Processing)
        # -----------------------------------------------------
        self.primary_results["global"] = {
            "K_global": K_global,
            "F_global": F_global,
            "K_mod": K_mod,
            "F_mod": F_mod,
            "K_cond": K_cond,
            "F_cond": F_cond,
            "U_cond": U_cond,
            "R_cond": R_cond,  # For debugging purposes
            "U_global": U_global,
            "R_global": R_global,
        }
        logger.info("Stored initial global results (pre-processing).")

        # -----------------------------------------------------
        # POST-PROCESSING FUNCTION: Updates Global Results
        # -----------------------------------------------------
        def post_processing():
            """
            Computes nodal rotations (θ_z) and bending moments (M_z) for Euler-Bernoulli beam elements,
            updates U_global and F_mod, and recalculates R_global_updated.
            """
            logger.info("Post-processing global results for Euler-Bernoulli beam elements...")

            # Retrieve element lengths and element types using dictionary keys.
            element_lengths = self.settings["mesh_dictionary"]["element_lengths"]
            element_types = self.settings["mesh_dictionary"]["element_types"]

            # Ensure geometry_array is 1D.
            geometry_array = self.settings["geometry_array"]
            if geometry_array.ndim > 1:
                geometry_array = geometry_array.flatten()
            logger.debug("Geometry array used for post-processing: shape %s", geometry_array.shape)

            U_global_updated = U_global.copy()
            F_mod_updated = F_mod.copy()
            num_nodes = len(U_global) // 6  # assuming 6 DOFs per node
            logger.info("Number of nodes: %d", num_nodes)

            # Define historical indices.
            MATERIAL_E_INDEX = 0    # Young's Modulus index (historically)
            GEOMETRY_IZ_INDEX = 3   # Moment of inertia about z-axis index (historically)

            for element_id, element_type in enumerate(element_types):
                if element_type == "EulerBernoulliBeamElement3DOF":
                    # Extract E from material_array using 2D indexing (assumed shape (1, n)).
                    try:
                        E = float(self.settings["material_array"][0, MATERIAL_E_INDEX])
                    except Exception as ex:
                        logger.error("Error extracting E for element %d: %s", element_id, ex)
                        raise

                    # Extract I_z from geometry_array using 2D indexing.
                    if geometry_array.size > GEOMETRY_IZ_INDEX:
                        I_z = float(self.settings["geometry_array"][0, GEOMETRY_IZ_INDEX])
                    else:
                        logger.warning("geometry_array has size %d; expected at least %d. Using geometry_array[0,0] for I_z.",
                                       geometry_array.size, GEOMETRY_IZ_INDEX+1)
                        I_z = float(self.settings["geometry_array"][0, 0])
                    
                    EI = E * I_z

                    # Get L for the element.
                    try:
                        L = float(element_lengths[element_id])
                    except Exception as ex:
                        logger.error("Error extracting L for element %d: %s", element_id, ex)
                        raise

                    logger.debug("Element %d: E=%.3e, I_z=%.3e, L=%.3e, EI=%.3e", element_id, E, I_z, L, EI)

                    theta_z_values = np.zeros(num_nodes)
                    Mz_values = np.zeros(num_nodes)

                    # Compute θ_z using a three-point one-sided stencil.
                    for i in range(num_nodes):
                        index = 6 * i + 1  # u_y index
                        if i == 0:
                            theta_z_values[i] = (-3 * U_global[index] + 4 * U_global[index + 6] - U_global[index + 12]) / (2 * L)
                        elif i == num_nodes - 1:
                            theta_z_values[i] = (3 * U_global[index] - 4 * U_global[index - 6] + U_global[index - 12]) / (2 * L)
                        else:
                            theta_z_values[i] = (U_global[index + 6] - U_global[index]) / L
                    logger.debug("Element %d: Computed theta_z_values: %s", element_id, theta_z_values)

                    # Compute M_z using a four-point one-sided stencil.
                    for i in range(num_nodes):
                        index = 6 * i + 1
                        if i == 0:
                            Mz_values[i] = EI * (2 * U_global[index] - 5 * U_global[index + 6] + 4 * U_global[index + 12] - U_global[index + 18]) / (L**2)
                        elif i == num_nodes - 1:
                            Mz_values[i] = EI * (2 * U_global[index] - 5 * U_global[index - 6] + 4 * U_global[index - 12] - U_global[index - 18]) / (L**2)
                        else:
                            Mz_values[i] = EI * (U_global[index + 6] - 2 * U_global[index] + U_global[index - 6]) / (L**2)
                    logger.debug("Element %d: Computed Mz_values: %s", element_id, Mz_values)

                    # Insert computed θ_z and M_z into updated global arrays.
                    for node_id in range(num_nodes):
                        dof_index = 6 * node_id + 5
                        U_global_updated[dof_index] = theta_z_values[node_id]
                        F_mod_updated[dof_index] = Mz_values[node_id]

            # Log differences between original and updated U_global and F_mod.
            diff_U = np.linalg.norm(U_global_updated - U_global)
            diff_F = np.linalg.norm(F_mod_updated - F_mod)
            logger.info("Post-processing differences: ||ΔU_global||=%.3e, ||ΔF_mod||=%.3e", diff_U, diff_F)

            # Update global results in primary_results.
            self.primary_results["global"]["U_global"] = U_global_updated
            self.primary_results["global"]["F_mod"] = F_mod_updated

            # Compute and store updated R_global_updated using bc_dofs.
            R_global_updated = np.zeros_like(F_mod_updated)
            R_global_updated[bc_dofs] = -F_mod_updated[bc_dofs]
            logger.info("Computed updated R_global with norm: %.3e", np.linalg.norm(R_global_updated))
            self.primary_results["global"]["R_global"] = R_global_updated

            return U_global_updated, F_mod_updated, R_global_updated

        # -----------------------------------------------------
        # Check if global results need updating (for Euler-Bernoulli beams)
        # -----------------------------------------------------
        if "EulerBernoulliBeamElement3DOF" in self.settings["mesh_dictionary"]["element_types"]:
            logger.info("Euler-Bernoulli beam elements detected; applying post-processing updates.")
            U_global_updated, F_mod_updated, R_global_updated = post_processing()
        else:
            logger.info("No Euler-Bernoulli beam elements detected; using original global results.")
            U_global_updated, F_mod_updated, R_global_updated = U_global, F_mod, R_global

        # -----------------------------------------------------
        # Compute element-wise results (after global updates)
        # -----------------------------------------------------
        elements = self.elements  # List of element objects

        try:
            K_e_mod, F_e_mod, U_e, R_e = disassemble_global_matrices(
                elements, K_mod, F_mod_updated, U_global_updated, R_global_updated
            )
        except ValueError as e:
            logger.error("Error during disassembly of element-wise results: %s", e)
            return None, None

        # Store element-wise results.
        self.primary_results["element"] = {"data": [
            {"element_id": i, "K_e_mod": K_e_mod[i], "F_e_mod": F_e_mod[i], "U_e": U_e[i], "R_e": R_e[i]}
            for i in range(len(K_e_mod))
        ]}
        logger.info("Finalized primary results with %d element-wise entries.", len(K_e_mod))

        return self.primary_results["global"], self.primary_results["element"]["data"]
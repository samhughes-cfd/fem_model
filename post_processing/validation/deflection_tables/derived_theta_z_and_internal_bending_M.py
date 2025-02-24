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
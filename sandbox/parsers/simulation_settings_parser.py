import os
import logging
import re

class SimulationSettingParser:
    def __init__(self, config_file):
        self.config_file = config_file

    def parse_simulation_type(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"[Simulation Settings] File not found: {self.config_file}")

        logging.info(f"[Simulation Settings] Reading file: {self.config_file}")

        solver_section_found = False
        sim_type_section_found = False
        selected_simulation_type = None

        with open(self.config_file, 'r') as f:
            for line_number, raw_line in enumerate(f, 1):
                line = raw_line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                if not solver_section_found:
                    if re.match(r"\[Solver\]", line, re.IGNORECASE):
                        solver_section_found = True
                        continue
                elif not sim_type_section_found:
                    if re.match(r"\[simulation_type\]", line, re.IGNORECASE):
                        sim_type_section_found = True
                        continue
                else:
                    # We are inside the simulation_type section
                    if line.startswith('#'):
                        continue
                    selected_simulation_type = line.strip().lower()
                    logging.info(f"[Simulation Settings] Selected simulation type: {selected_simulation_type}")
                    break

        if not selected_simulation_type:
            raise ValueError("[Simulation Settings] No simulation type selected. Please uncomment one option under [simulation_type].")

        if selected_simulation_type not in {'static', 'dynamic', 'modal'}:
            raise ValueError(f"[Simulation Settings] Unknown simulation type: '{selected_simulation_type}'")

        return selected_simulation_type
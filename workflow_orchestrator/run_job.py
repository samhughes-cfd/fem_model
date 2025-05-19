# run_job.py

import os
import sys
import glob
import logging
import time
import numpy as np
import multiprocessing as mp
import psutil
import shutil
import platform
import datetime
import uuid
from tabulate import tabulate
import cpuinfo
import subprocess

# Adjust Python Path to include project root
script_dir = os.path.dirname(os.path.abspath(__file__))
fem_model_root = os.path.abspath(os.path.join(script_dir, '..'))
if fem_model_root not in sys.path:
    sys.path.insert(0, fem_model_root)

# Import required modules
from pre_processing.parsing.geometry_parser import parse_geometry
from pre_processing.parsing.mesh_parser import parse_mesh
from pre_processing.parsing.material_parser import parse_material
from pre_processing.parsing.solver_parser import parse_solver
from pre_processing.parsing.point_load_parser import parse_point_load
from pre_processing.parsing.distributed_load_parser import parse_distributed_load
from processing.solver_registry import get_solver_registry
from simulation_runner.static.static_simulation import StaticSimulationRunner
from pre_processing.element_library.element_factory import create_elements_batch

# Configure logging for the main process
def configure_logging(log_file_path):
    """Configures logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
        ]
    )
    logger = logging.getLogger(__name__)
    logger.propagate = False
    return logger

def configure_child_logging(job_results_dir):
    """Configures logging for a child process."""
    log_file_path = os.path.join(job_results_dir, "process_job.log")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return logger

def get_machine_specs():
    """Returns extended system specifications as a formatted string."""
    cpu_info = cpuinfo.get_cpu_info()
    cpu_name = cpu_info.get('brand_raw', platform.processor())
    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    total_ram = psutil.virtual_memory().total / (1024 ** 3)
    disk_total, _, disk_free = shutil.disk_usage("/")

    specs = (
        f"Machine Specifications:\n"
        f"   - OS: {platform.system()} {platform.release()} ({platform.version()})\n"
        f"   - CPU: {cpu_name}\n"
        f"       • Logical cores: {logical_cores}\n"
        f"       • Physical cores: {physical_cores}\n"
        f"   - RAM: {total_ram:.2f} GB\n"
        f"   - Disk: {disk_total / (1024 ** 3):.2f} GB total, {disk_free / (1024 ** 3):.2f} GB free\n"
        f"   - Python Version: {platform.python_version()} ({sys.executable})\n"
    )

    try:
        result = subprocess.check_output("wmic path win32_VideoController get name", shell=True)
        gpus = result.decode().split('\n')[1:]
        gpus = [g.strip() for g in gpus if g.strip()]
        specs += f"   - GPU(s): {', '.join(gpus)}\n"
    except Exception:
        specs += "   - GPU(s): Unable to detect (requires Windows & WMIC)\n"

    return specs

def track_usage():
    """Returns current memory, disk, and CPU usage."""
    process = psutil.Process(os.getpid())
    return {
        "Memory (MB)": process.memory_info().rss / (1024 * 1024),
        "Disk (GB)": psutil.disk_usage('/').used / (1024 ** 3),
        "CPU (%)": process.cpu_percent(interval=0.1)
    }

def process_job(job_dir, job_times, job_start_end_times, base_settings):
    """Processes a single FEM simulation job."""
    case_name = os.path.basename(job_dir)
    # Modified directory name with microsecond timestamp and unique identifier
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    unique_id = uuid.uuid4().hex[:8]  # 8-character random string
    job_results_dir = os.path.join(
        fem_model_root, 
        'post_processing', 
        'results', 
        f"{case_name}_{timestamp}_pid{os.getpid()}_{unique_id}"
    )
    os.makedirs(job_results_dir, exist_ok=True)

    logger = configure_child_logging(job_results_dir)
    logger.info(f"🟢 Starting job: {case_name}")

    start_time = time.time()
    usage_start = track_usage()
    performance_log_path = os.path.join(job_results_dir, "job_performance.log")
    performance_data = [["Step", "Time (s)", "Memory (MB)", "Disk (GB)", "CPU (%)"]]

    try:
        # --- Parsing Input Files ---
        step_start = time.time()
        mesh_dictionary = parse_mesh(os.path.join(job_dir, "mesh.txt"))
        point_load_array = np.array([])
        distributed_load_array = np.array([])
        
        point_load_path = os.path.join(job_dir, "point_load.txt")
        if os.path.exists(point_load_path):
            point_load_array = parse_point_load(point_load_path)
        
        distributed_load_path = os.path.join(job_dir, "distributed_load.txt")
        if os.path.exists(distributed_load_path):
            distributed_load_array = parse_distributed_load(distributed_load_path)
        
        parsing_time = time.time() - step_start
        performance_data.append(["Parsing", parsing_time, *track_usage().values()])

        # --- Element Instantiation ---
        step_start = time.time()
        element_ids = mesh_dictionary["element_ids"]
        
        # Add unique job identifier to parameters
        params_list = np.array([{
            "geometry_array": base_settings["geometry"],
            "material_array": base_settings["material"],
            "mesh_dictionary": mesh_dictionary,
            "point_load_array": point_load_array,
            "distributed_load_array": distributed_load_array,
            "element_id": f"{elem_id}_{unique_id}",  # Unique element identifier
            "job_results_dir": job_results_dir
        } for elem_id in element_ids], dtype=object)

        all_elements = create_elements_batch(mesh_dictionary, params_list)

        if any(elem is None for elem in all_elements):
            logger.error(f"❌ Error: Some elements failed to instantiate in {case_name}.")
            raise ValueError(f"❌ Invalid elements detected in {case_name}.")
        
        # Verify logging directories were created
        required_dirs = [
            os.path.join(job_results_dir, "element_stiffness_matrices"),
            os.path.join(job_results_dir, "element_force_vectors")
        ]
        for d in required_dirs:
            if not os.path.exists(d):
                logger.error(f"Missing required directory: {d}")
                raise FileNotFoundError(f"Directory not created: {d}")

        element_creation_time = time.time() - step_start
        performance_data.append(["Element Instantiation", element_creation_time, *track_usage().values()])

        # --- Element Computations ---
        # 1) Stiffness matrices
        step_start = time.time()
        vectorized_stiffness = np.vectorize(
            lambda elem: elem.element_stiffness_matrix() if elem else None,
            otypes=[object]
        )
        element_stiffness_matrices = vectorized_stiffness(all_elements)
        stiffness_time = time.time() - step_start
        performance_data.append(["Element Stiffness Computation", stiffness_time, *track_usage().values()])

        # 2) Force vectors
        step_start = time.time()
        vectorized_force = np.vectorize(
            lambda elem: elem.element_force_vector() if elem else None,
            otypes=[object]
        )
        element_force_vectors = vectorized_force(all_elements)
        force_time = time.time() - step_start
        performance_data.append(["Element Force Computation", force_time, *track_usage().values()])

        # --- Simulation Setup ---
        runner = StaticSimulationRunner(
            settings={
                "elements": all_elements,
                "mesh_dictionary": mesh_dictionary,
                "material_array": base_settings["material"],
                "geometry_array": base_settings["geometry"],
                "solver_name": base_settings["solver"][0],
                "element_stiffness_matrices": element_stiffness_matrices,
                "element_force_vectors": element_force_vectors
            },
            job_name=case_name
        )

        # --- Simulation Steps with Path Passing ---
        step_start = time.time()
        runner.setup_simulation()
        setup_time = time.time() - step_start
        performance_data.append(["Setup Simulation", setup_time, *track_usage().values()])

        step_start = time.time()
        K_global, F_global = runner.assemble_global_matrices(job_results_dir)
        assembly_time = time.time() - step_start
        performance_data.append(["Assemble Global Matrices", assembly_time, *track_usage().values()])

        step_start = time.time()
        K_mod, F_mod, fixed_dofs = runner.modify_global_matrices(K_global, F_global, job_results_dir)
        modify_time = time.time() - step_start
        performance_data.append(["Modify Global Matrices", modify_time, *track_usage().values()])

        step_start = time.time()
        U_global, K_cond, F_cond, U_cond = runner.solve_static(K_mod, F_mod, fixed_dofs, job_results_dir)
        solve_time = time.time() - step_start
        performance_data.append(["Solve Linear Static System", solve_time, *track_usage().values()])

        step_start = time.time()
        runner.compute_primary_results(K_global, F_global, K_mod, F_mod, K_cond, F_cond, U_cond, U_global)
        compute_primary_time = time.time() - step_start
        performance_data.append(["Compute Primary Results", compute_primary_time, *track_usage().values()])

        step_start = time.time()
        runner.save_primary_results(output_dir=job_results_dir)
        save_primary_time = time.time() - step_start
        performance_data.append(["Save Primary Results", save_primary_time, *track_usage().values()])

        # --- Final Tracking ---
        end_time = time.time()
        usage_end = track_usage()
        job_times[case_name] = {"total_time": end_time - start_time}
        job_start_end_times[case_name] = (start_time, end_time)

        parallel_jobs = [
            job for job, (s, e) in job_start_end_times.items()
            if (s < end_time and e > start_time) and job != case_name
        ]

        with open(performance_log_path, "w") as f:
            f.write(get_machine_specs() + "\n")
            f.write(f"Job: {case_name}\n")
            f.write(f"Timestamp (job start): {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}\n")
            f.write(f"Total Time: {end_time - start_time:.2f} sec\n\n")
            f.write(tabulate(performance_data, headers="firstrow", tablefmt="grid") + "\n\n")
            f.write(f"Parallel Jobs: {', '.join(parallel_jobs) if parallel_jobs else 'None'}\n")
            f.write(f"Start Memory: {usage_start['Memory (MB)']:.2f} MB | End Memory: {usage_end['Memory (MB)']:.2f} MB\n")
            f.write(f"Start Disk Usage: {usage_start['Disk (GB)']:.2f} GB | End Disk Usage: {usage_end['Disk (GB)']:.2f} GB\n")
            f.write(f"Start CPU: {usage_start['CPU (%)']:.2f}% | End CPU: {usage_end['CPU (%)']:.2f}%\n")

    except Exception as e:
        logger.error(f"❌ Error in job {case_name}: {e}", exc_info=True)
    
        # Close all log handlers to release file locks
        for handler in logger.handlers:
            handler.close()
        logging.shutdown()
    
        # Retry cleanup with error handling
        try:
            if os.path.exists(job_results_dir):
                shutil.rmtree(job_results_dir, ignore_errors=False)
        except Exception as cleanup_err:
            logger.error(f"❌ Cleanup failed for {job_results_dir}: {cleanup_err}")

def main():
    """Manages and runs multiple FEM simulation jobs in parallel."""
    log_file_path = os.path.join(script_dir, "run_job.log")
    logger = configure_logging(log_file_path)
    logger.info("🚀 Starting FEM Simulation Workflow")

    jobs_dir = os.path.join(fem_model_root, 'jobs')
    base_dir = os.path.join(jobs_dir, "base")
    job_dirs = [d for d in glob.glob(os.path.join(jobs_dir, 'job_*')) if os.path.isdir(d)]
    
    if not job_dirs:
        logger.warning("⚠️ No job directories found.")
        return

    logger.info("📥 Loading base settings...")
    base_settings = {
        "geometry": parse_geometry(os.path.join(base_dir, "geometry.txt")),
        "material": parse_material(os.path.join(base_dir, "material.txt")),
        "solver": parse_solver(os.path.join(base_dir, "solver.txt")),
    }
    logger.info("✅ Base settings loaded successfully.")

    num_processes = min(mp.cpu_count(), len(job_dirs))
    logger.info(f"🟢 Running {len(job_dirs)} jobs across {num_processes} CPU cores.")

    with mp.Manager() as manager:
        job_times = manager.dict()
        job_start_end_times = manager.dict()

        with mp.Pool(processes=num_processes) as pool:
            pool.starmap(
                process_job,
                [(job, job_times, job_start_end_times, base_settings) for job in job_dirs]
            )

if __name__ == "__main__":
    main()
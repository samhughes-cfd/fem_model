import os
import numpy as np
import matplotlib.pyplot as plt
import glob

def find_force_files(job_dir):
    """
    Finds force vector files in primary_results.
    """
    primary_results_dir = os.path.join(job_dir, "primary_results")
    
    filename_patterns = {
        "element_F": "static_element_F_e_*.txt",
        "global_F": "static_global_F_global_*.txt",
        "global_F_mod": "static_global_F_mod_*.txt",
        "global_F_cond": "static_global_F_cond_*.txt"
    }
    
    resolved_files = {}
    print("\n🔍 Searching for force vector files...")
    
    for key, pattern in filename_patterns.items():
        search_pattern = os.path.join(primary_results_dir, f"*{pattern}")
        matched_files = sorted(glob.glob(search_pattern))
        
        if matched_files:
            resolved_files[key] = matched_files[-1]
            print(f"✔ Found: {resolved_files[key]}")
        else:
            print(f"❌ No matching files found for: {pattern}")
    
    return resolved_files

def parse_element_force(file_path):
    """
    Parses element force vector file.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    forces = []
    element_id = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("# Element ID:"):
            element_id = int(line.split(":")[-1].strip())
        elif line and not line.startswith("#"):
            values = list(map(float, line.split(',')))
            forces.append((element_id, values))
    
    return forces

def parse_global_force(file_path):
    """
    Parses global force vector file.
    """
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines() if line.strip() and not line.startswith("#")]
    
    return np.array([float(val) for val in lines])

def write_force_vectors(force_data, output_dir):
    """
    Writes force vectors to structured .txt files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for key, data in force_data.items():
        output_file = os.path.join(output_dir, f"{key}.txt")
        
        with open(output_file, 'w') as f:
            f.write(f"Force Vector: {key}\n\n")
            
            if isinstance(data, list):  # Element force data
                for element_id, forces in data:
                    f.write(f"Element {element_id}: {', '.join(f'{v:.2e}' for v in forces)}\n")
            else:  # Global force data
                f.write('\n'.join(f"{v:.2e}" for v in data))
    
    print(f"✅ Force vectors saved in: {output_dir}")

def plot_force_distribution(force_data, title, output_dir):
    """
    Plots a histogram of force magnitudes.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if isinstance(force_data, list):
        values = [abs(v) for _, forces in force_data for v in forces]
    else:
        values = np.abs(force_data)
    
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, log=True, edgecolor='black')
    
    plt.xlabel("Force Magnitude")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Force Magnitudes: {title}")
    
    output_file = os.path.join(output_dir, f"{title}_distribution.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Force distribution plot saved to {output_file}")

def load_and_process_force_vectors(job_dir, output_base_dir):
    """
    Finds, parses, writes, and plots force vectors.
    """
    force_files = find_force_files(job_dir)
    force_vectors = {}
    
    job_name = os.path.basename(job_dir)
    output_dir = os.path.join(output_base_dir, "force_vector_evolution", job_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("📁 Created output directory.")
    
    for key, file_path in force_files.items():
        if os.path.exists(file_path):
            print(f"📖 Reading: {file_path}")
            if "element_F" in key:
                force_vectors[key] = parse_element_force(file_path)
            else:
                force_vectors[key] = parse_global_force(file_path)
        else:
            print(f"⚠ Skipping missing file: {file_path}")
    
    if not force_vectors:
        print("❌ No force vectors found. Exiting...")
        return
    
    write_force_vectors(force_vectors, output_dir)
    
    for key, data in force_vectors.items():
        plot_force_distribution(data, key, output_dir)
    
    print(f"\n✅ Force processing complete. Results saved in: {output_dir}")
    return force_vectors

if __name__ == "__main__":
    base_directory = "post_processing/results"
    output_directory = "post_processing/tensor_visualisers"
    selected_job = "job_0003_2025-02-26_16-02-28"
    
    job_directory = os.path.join(base_directory, selected_job)
    
    if os.path.exists(job_directory):
        load_and_process_force_vectors(job_directory, output_directory)
    else:
        print(f"Error: Job directory {job_directory} does not exist.")

# In this experiment, we will test the performance of our Jetson Board 
# in different power modes while running a computationally intensive task. 
# We will use a gemm file to measure performance and energy consumption in each mode.

import os
import sys
import subprocess
import shutil

# Ensure we can import from the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from scripts.jetson_tools import modify_power_mode

def get_available_modes():
    """
    Reads the NVIDIA power model configuration file to find all 
    available power mode IDs for the current Jetson device.
    """
    modes = []
    try:
        with open('/etc/nvpmodel.conf', 'r') as f:
            for line in f:
                if line.strip().startswith('< POWER_MODEL'):
                    # Example format: < POWER_MODEL ID=0 NAME=MAXN >
                    parts = line.split()
                    for part in parts:
                        if part.startswith('ID='):
                            modes.append(int(part.split('=')[1]))
    except FileNotFoundError:
        print("Warning: /etc/nvpmodel.conf not found. Defaulting to test modes 0-8.")
        return list(range(9))
    
    # Return a sorted list of unique mode IDs
    return sorted(list(set(modes)))

def run_experiment():
    print("=" * 70)
    print("Starting Power Mode Sweep Experiment")
    print("=" * 70)
    
    modes = get_available_modes()
    if not modes:
        print("No power modes found. Ensure you are running on a Jetson device.")
        return

    print(f"Discovered available power modes: {modes}")

    # The gemm.py script is located in the parent directory
    gemm_script = os.path.join(parent_dir, "gemm.py")

    for mode in modes:
        print(f"\n{'=' * 50}")
        print(f"Testing Power Mode: {mode}")
        print(f"{'=' * 50}")
        
        # 1. Change power mode using the function from jetson_tools
        modify_power_mode(mode)
        
        # Double check if the mode was successfully applied
        exit_code = os.system(f"sudo nvpmodel -m {mode} > /dev/null 2>&1")
        if exit_code != 0:
            print(f"Skipping mode {mode} as it failed to set. Are you running with sudo?")
            continue

        # 2. Clean up previous artifacts to ensure `gemm.py` merges correctly.
        # Since `energy_results.csv` appends by default, we MUST delete it between
        # runs so that `merge_csvs_by_row_order` maps the rows perfectly.
        for temp_file in ["model_benchmarks.csv", "energy_results.csv", "bench_with_energy.csv"]:
            temp_path = os.path.join(parent_dir, temp_file)
            if os.path.exists(temp_path):
                os.remove(temp_path)

        print(f"\nRunning benchmark for mode {mode}...")
        
        # 3. Run gemm.py
        # It natively handles the @measure_energy_to_csv decorator
        result = subprocess.run(["python3", gemm_script], cwd=parent_dir)
        
        if result.returncode != 0:
            print(f"Error: Benchmark failed for mode {mode}.")
            continue
            
        # 4. Save the merged results with a specific mode identifier
        source_csv = os.path.join(parent_dir, "bench_with_energy.csv")
        target_csv = os.path.join(parent_dir, f"bench_with_energy_mode_{mode}.csv")
        
        if os.path.exists(source_csv):
            shutil.copy(source_csv, target_csv)
            print(f"\n✅ Saved mode {mode} results to: {target_csv}")
        else:
            print(f"\n⚠️ Warning: {source_csv} was not generated.")

    print("\nExperiment complete! Check the parent directory for your CSV files.")

if __name__ == "__main__":
    # Nvpmodel requires root privileges to scale clocks and toggle cores
    if os.geteuid() != 0:
        print("ERROR: Please run this script with sudo to allow nvpmodel to change power modes.")
        print("Usage: sudo python3 example_experiment/jetson_npmode.py")
        sys.exit(1)
        
    run_experiment()

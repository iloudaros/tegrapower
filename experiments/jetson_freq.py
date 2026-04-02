import os
import sys
import subprocess
import shutil

# Ensure we can import from the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from scripts.jetson_tools import modify_gpu_freq

def get_available_gpu_freqs():
    """
    Determines the correct sysfs path for the current Jetson model
    and reads the available GPU frequencies.
    """
    try:
        model = os.popen("tr -d '\\0' < /proc/device-tree/model").read().strip()
    except Exception as e:
        print(f"Error reading model: {e}")
        return []

    if "Jetson Nano" in model:
        path = '/sys/devices/57000000.gpu/devfreq/57000000.gpu'
    elif "Xavier NX" in model or "AGX" in model:
        path = '/sys/devices/17000000.gv11b/devfreq/17000000.gv11b'
    else:
        print(f"Model not supported or recognized for frequency sweep: {model}")
        return []

    try:
        with open(f'{path}/available_frequencies', 'r') as file:
            # Read frequencies, convert to integers, and sort them
            freqs = sorted([int(f) for f in file.read().split()])
            return freqs
    except FileNotFoundError:
        print(f"Error: Could not find GPU frequency files at {path}.")
        return []

def run_frequency_experiment():
    print("=" * 70)
    print("Starting GPU Frequency Sweep Experiment")
    print("=" * 70)
    
    # Optional: You might want to ensure the board is in a high-power mode 
    # (like MAXN) before sweeping frequencies so power-capping doesn't interfere.
    # from scripts.jetson_tools import modify_power_mode
    # modify_power_mode(0)  

    frequencies = get_available_gpu_freqs()
    if not frequencies:
        print("No GPU frequencies found. Ensure you are running on a supported Jetson device.")
        return

    print(f"Discovered {len(frequencies)} available GPU frequencies (Hz):")
    print(frequencies)

    gemm_script = os.path.join(parent_dir, "gemm.py")

    for freq in frequencies:
        print(f"\n{'=' * 50}")
        # Convert Hz to MHz for easier reading in the console
        print(f"Testing GPU Frequency: {freq} Hz ({freq / 1_000_000:.2f} MHz)")
        print(f"{'=' * 50}")
        
        # 1. Change GPU frequency using the function from jetson_tools
        modify_gpu_freq(freq)

        # 2. Clean up previous artifacts to ensure `gemm.py` merges correctly.
        for temp_file in ["model_benchmarks.csv", "energy_results.csv", "bench_with_energy.csv"]:
            temp_path = os.path.join(parent_dir, temp_file)
            if os.path.exists(temp_path):
                os.remove(temp_path)

        print(f"\nRunning benchmark for frequency {freq} Hz...")
        
        # 3. Run gemm.py
        result = subprocess.run(["python3", gemm_script], cwd=parent_dir)
        
        if result.returncode != 0:
            print(f"Error: Benchmark failed for frequency {freq}.")
            continue
            
        # 4. Save the merged results with a specific frequency identifier
        source_csv = os.path.join(parent_dir, "bench_with_energy.csv")
        target_csv = os.path.join(parent_dir, f"bench_with_energy_freq_{freq}.csv")
        
        if os.path.exists(source_csv):
            shutil.copy(source_csv, target_csv)
            print(f"\n✅ Saved frequency {freq} Hz results to: {target_csv}")
        else:
            print(f"\n⚠️ Warning: {source_csv} was not generated.")

    print("\nExperiment complete! Check the parent directory for your CSV files.")

if __name__ == "__main__":
    # Writing to sysfs max_freq/min_freq requires root privileges
    if os.geteuid() != 0:
        print("ERROR: Please run this script with sudo to allow modifying GPU frequencies.")
        print("Usage: sudo python3 example_experiment/jetson_freq.py")
        sys.exit(1)
        
    run_frequency_experiment()

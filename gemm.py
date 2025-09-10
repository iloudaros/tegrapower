import torch
import os
import time
import sys
import csv
from tegrapower import measure_energy_to_csv, merge_csvs_by_row_order, load_dot_env, extract_env_var

# Load environment variables from .env file if present
load_dot_env()

# ==============================================================================
#  1. Define the base matrix shapes (I, J, K) from model architectures
#     - A (I, K) matrix is multiplied by a (K, J) matrix.
# ==============================================================================

BASE_CONFIGS = [
    {'model': 'MISC_2', 'I': 512, 'J': 896,  'K': 896,  'runs': 100},
    {'model': 'MISC_2', 'I': 512, 'J': 128,  'K': 896,  'runs': 100},
    {'model': 'MISC_2', 'I': 512, 'J': 4864, 'K': 896,  'runs': 100},
    {'model': 'MISC_2', 'I': 512, 'J': 896,  'K': 4896, 'runs': 100},
    # ------------------------------------------------------------------------ #
    {'model': 'MISC_2', 'I': 512, 'J': 2048, 'K': 2048, 'runs': 100},
    {'model': 'MISC_2', 'I': 512, 'J': 512,  'K': 2048, 'runs': 100},
    {'model': 'MISC_2', 'I': 512, 'J': 8192, 'K': 2048, 'runs': 100},
    {'model': 'MISC_2', 'I': 512, 'J': 2048, 'K': 8192, 'runs': 100},
    # ------------------------------------------------------------------------ #
    {'model': 'MISC_2', 'I': 224, 'J': 768,  'K': 768,  'runs': 100},
    {'model': 'MISC_2', 'I': 224, 'J': 224,  'K': 64,   'runs': 100},
    {'model': 'MISC_2', 'I': 224, 'J': 3072, 'K': 768,  'runs': 100},
    {'model': 'MISC_2', 'I': 224, 'J': 768,  'K': 3072, 'runs': 100},
    # ------------------------------------------------------------------------ #
    {'model': 'MISC_2', 'I': 224, 'J': 96,   'K': 96,   'runs': 100},
    {'model': 'MISC_2', 'I': 224, 'J': 64,   'K': 224,  'runs': 100},
    {'model': 'MISC_2', 'I': 224, 'J': 384,  'K': 96,   'runs': 100},
    {'model': 'MISC_2', 'I': 224, 'J': 96,   'K': 384,  'runs': 100},
    # ------------------------------------------------------------------------ #
    {'model': 'MISC_2', 'I': 288, 'J': 1536, 'K': 1536, 'runs': 100},
    {'model': 'MISC_2', 'I': 288, 'J': 288,  'K': 64,   'runs': 100},
    {'model': 'MISC_2', 'I': 288, 'J': 64,   'K': 288,  'runs': 100},
    {'model': 'MISC_2', 'I': 288, 'J': 6144, 'K': 1536, 'runs': 100},
    {'model': 'MISC_2', 'I': 288, 'J': 1536, 'K': 6144, 'runs': 100},
    # ------------------------------------------------------------------------ #
    {'model': 'MISC_2', 'I': 512, 'J': 1536, 'K': 1536, 'runs': 100},
    {'model': 'MISC_2', 'I': 512, 'J': 512,  'K': 192,  'runs': 100},
    {'model': 'MISC_2', 'I': 512, 'J': 192,  'K': 512,  'runs': 100},
    {'model': 'MISC_2', 'I': 512, 'J': 6144, 'K': 1536, 'runs': 100},
    {'model': 'MISC_2', 'I': 512, 'J': 1536, 'K': 6144, 'runs': 100},
]


# ==============================================================================
#  2. Define the batch sizes to apply to EVERY shape above
# ==============================================================================
#BATCH_SIZES = [1, 4, 8, 16]
BATCH_SIZES = [1]


# ==============================================================================
#  3. Define the output file name
# ==============================================================================
OUTPUT_CSV_FILE = 'model_benchmarks.csv'


@measure_energy_to_csv(
    rail=extract_env_var("rail", list, default="VIN_SYS_5V0"),            
    interval_ms=extract_env_var("interval_ms", int, default=50),
    log_dir=extract_env_var("log_dir", str, default="powerlogs"),              
    num_runs=extract_env_var("num_runs", int, default=5),                       
    guard_samples=extract_env_var("guard_samples", int, default=3),                  
    energy_csv_path=extract_env_var("energy_csv_path", str, default="energy_results.csv"),
    append=extract_env_var("append", bool, default=True),
    also_write_log_file=extract_env_var("also_write_log_file", bool, default=True),
    fallback_rails=extract_env_var("fallback_rails", list, default=["POM_5V_IN", "VDD_CPU_CV", "VDD_GPU_SOC"]),
)    
def run_benchmark(op_func, runs):
    """
    Benchmarks an operation and returns average latency in seconds.
    Ensures the measured segment is long enough for reliable energy capture.
    """
    min_segment_s = 1.0  # ensure enough samples for very short ops

    try:
        # Warm-up runs
        for _ in range(10):
            op_func()
        torch.cuda.synchronize()

        # Estimate single-run latency to decide repeats
        torch.cuda.synchronize()
        t0 = time.time()
        op_func()
        torch.cuda.synchronize()
        t1 = time.time()
        single_run_s = max(1e-6, t1 - t0)

        repeats = max(runs, int(min_segment_s / single_run_s))

        start_time = time.time()
        for _ in range(repeats):
            op_func()
        torch.cuda.synchronize()
        end_time = time.time()

        return (end_time - start_time) / repeats

    except torch.cuda.OutOfMemoryError:
        print("    -> ERROR: CUDA out of memory. Skipping.")
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"    -> ERROR running benchmark: {e}")
        return None


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    dtype = torch.float32

    print(f"PyTorch version: {torch.__version__}")
    print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    print(f"Data Type: {dtype}")
    print(f"Results will be saved to: {OUTPUT_CSV_FILE}")
    print("=" * 70)

    with open(OUTPUT_CSV_FILE, 'w', newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['model', 'I', 'J', 'K', 'BATCH_SIZE', 'throughput_gops', 'latency_sec']
        csv_writer.writerow(header)

        total_tests = len(BASE_CONFIGS) * len(BATCH_SIZES)
        current_test = 0

        for config in BASE_CONFIGS:
            model, I, K, J, runs = config['model'], config['I'], config['K'], config['J'], config['runs']

            for batch_size in BATCH_SIZES:
                current_test += 1
                print(f"\n--- Test {current_test}/{total_tests} ---")

                if batch_size == 1:
                    print(f"  Model: {model} | Type: GEMM | Shape: (I={I}, K={K}) @ (K={K}, J={J})")
                    lhs = torch.randn(I, K, device=device, dtype=dtype)
                    rhs = torch.randn(K, J, device=device, dtype=dtype)
                    op_func = lambda: torch.matmul(lhs, rhs)
                else:
                    print(f"  Model: {model} | Type: BMM  | Batch: {batch_size} | Shape: ({I},{K}) @ ({K},{J})")
                    lhs = torch.randn(batch_size, I, K, device=device, dtype=dtype)
                    rhs = torch.randn(batch_size, K, J, device=device, dtype=dtype)
                    op_func = lambda: torch.bmm(lhs, rhs)

                # Provide a clear tag so energy CSV has descriptive names
                tag = f"{model}_I{I}_K{K}_J{J}_B{batch_size}_runs{runs}"
                latency_sec = run_benchmark(op_func, runs, _bench_tag=tag)

                if latency_sec is not None:
                    # FLOPs = 2 * Batch * I * J * K
                    flops = 2.0 * batch_size * I * J * K
                    throughput_gops = (flops / latency_sec) / 1e9

                    print(f"    -> Avg Latency: {latency_sec:.6f} sec")
                    print(f"    -> Throughput:  {throughput_gops:.2f} GOPS")

                    row = [model, I, J, K, batch_size, f"{throughput_gops:.2f}", f"{latency_sec:.6f}"]
                    csv_writer.writerow(row)

                # Cleanup
                del lhs, rhs, op_func
                torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"Benchmark complete. Results saved in '{OUTPUT_CSV_FILE}'.")
    print("Energy per test saved in 'energy_results.csv' with columns [Test, Energy_J, Avg_power_mW].")
    merge_csvs_by_row_order("model_benchmarks.csv", "energy_results.csv", "bench_with_energy.csv")
    print("Merged results with energy data saved in 'bench_with_energy.csv'.")


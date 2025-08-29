# Measuring Energy on Jetson Devices with tegrapower.py

## Overview

`tegrapower.py` provides tools to measure power and energy on NVIDIA Jetson devices (e.g., AGX Orin) using `tegrastats`:

-  `TegrastatsLogger`: start/stop `tegrastats` and log to a file
-  Parsers and energy integration utilities
-  CSV exporters (time-series and summary)
-  Decorator `measure_energy_to_csv` to wrap a benchmark function:
   - Captures power during execution
   - Integrates energy for a chosen rail (e.g., `VIN_SYS_5V0`) or a sum of multiple rails (e.g., `["VDD_GPU_SOC", "VDD_CPU_CV"]`)
   - Appends results to a simple CSV: `[Test, Energy_J, Avg_Power_mW]`
-  Helper `merge_csvs_by_row_order` to merge benchmark and energy CSVs by row position

This README explains how to use `tegrapower.py` to measure energy for a GEMM benchmark similar to `gemm.py`, and how to use the provided Makefile to simplify the container workflow.

---

## Requirements

Mount both your workspace and the `tegrastats` binary into the container.

You can launch the container directly:

```bash
docker run -it --rm --gpus all --runtime nvidia --network host \
  -v /home/iloudaros/pytorch_eval:/workspace \
  -v /usr/bin/tegrastats:/usr/bin/tegrastats \
  nvcr.io/nvidia/l4t-ml:r36.2.0-py3
```

Or, use the configurable Makefile (recommended; see “Using the Makefile” below).

Inside the container:

-  Ensure `/workspace` contains your `tegrapower.py`, `gemm.py`, and `Makefile`
-  Ensure `tegrastats` is available at `/usr/bin/tegrastats` (mounted from the host)

### Notes about the Container

The example uses the NVIDIA L4T ML container `nvcr.io/nvidia/l4t-ml:r36.2.0-py3` which is compatible with AGX Orin. For other Jetson models, choose a suitable container from the [NVIDIA NGC catalog](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-ml). We provide recommendations below:


| Jetson Model      | Recommended Container                         |
|-------------------|-----------------------------------------------|
| Jetson AGX Orin   | nvcr.io/nvidia/l4t-ml:r36.2.0-py3             |
| Jetson AGX Xavier | nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 |
| Jetson NX Xavier  | nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 |

---

## Files

-  `tegrapower.py`: power logging, parsing, energy integration, CSV exporters, and decorator
-  `gemm.py`: example GEMM benchmark using the decorator
-  `Makefile`: convenience targets to query device info and launch the container with a configurable workspace

Place these files in your host workspace directory and mount it at `/workspace` inside the container.

---

## Using the Makefile

The Makefile makes the workflow simpler and supports a configurable workspace path.

Makefile (provided in the repo):

```makefile
# Configurable workspace path (host side)
# Override by:
#   - environment variable: export WORKSPACE=/path/to/your/workspace
#   - or at invocation: make docker WORKSPACE=/path/to/your/workspace
WORKSPACE ?= /home/iloudaros/pytorch_eval


# NVIDIA container image to use
# Override by:
#   - environment variable: export CONTAINER=your_preferred_image
#   - or at invocation: make docker CONTAINER=your_preferred_image
CONTAINER ?= nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# Read the Jetson model from device tree
model_val := $(shell tr -d '\0' < /proc/device-tree/model)

# Print detected Jetson model
model:
	@echo $(model_val)

# Show JetPack package information and L4T release details
jetpack_version:
	sudo apt-cache show nvidia-jetpack
	cat /etc/nv_tegra_release
	@echo ""
	@echo "Remember to check the Linux version mapping at:"
	@echo "https://docs.nvidia.com/jetson/archives/index.html"

# Launch the NVIDIA container with GPU access,
# mounting your configurable workspace and the tegrastats binary
docker:
	docker run -it --rm --gpus all --runtime nvidia --network host \
		-v $(WORKSPACE):/workspace \
		-v /usr/bin/tegrastats:/usr/bin/tegrastats \
		$(CONTAINER)

```

Run these from the host (not inside the container), in the directory containing the Makefile:

-  Configure the workspace path (host directory containing `tegrapower.py`, `gemm.py`, `Makefile`):
   - Option A (environment variable):
     - `export WORKSPACE=/path/to/your/workspace`
     - `make docker`
   - Option B (inline argument):
     - `make docker WORKSPACE=/path/to/your/workspace`
   - Default (if unset): `/home/iloudaros/pytorch_eval`

Targets:

-  Print device model:
   - `make model`
-  Show JetPack and L4T release info:
   - `make jetpack_version`
-  Launch the container:
   - `make docker` (with `WORKSPACE` configured)

Once inside the container, your files will be at `/workspace`.

---

## Quick Start (GEMM example)

### 1) Decorate your benchmark function

In `gemm.py`, the `run_benchmark` is decorated with `measure_energy_to_csv`. You can specify a single rail or a list of rails to sum their power and energy.

```python
from tegrapower import measure_energy_to_csv

# Example 1: Measure total system power from a single input rail
@measure_energy_to_csv(
    rail="VIN_SYS_5V0",                 # Primary rail to measure
    interval_ms=50,                     # Faster sampling for short tests
    log_dir="powerlogs",                # Where per-test tegrastats logs are stored
    guard_samples=3,                    # Pre/post guard samples (~0,15 s each at 50 ms)
    energy_csv_path="energy_results.csv",
    append=True,
    also_write_log_file=True,
    fallback_rails=["POM_5V_IN"] # Alternative if primary rail isn't found
)
def run_benchmark(op_func, runs):
    # ... benchmark logic ...

# Example 2: Sum the power of GPU and CPU rails
@measure_energy_to_csv(
    rail=["VDD_GPU_SOC", "VDD_CPU_CV"], # Sum these rails
    interval_ms=50,
    # ... other parameters ...
)
def run_benchmark(op_func, runs):
    # ... benchmark logic ...
```

Notes:

-  The `rail` parameter accepts a single string (e.g., `"VIN_SYS_5V0"`) or a list of strings (`["VDD_GPU_SOC", "VDD_CPU_CV"]`). When a list is provided, their power and energy values are summed for each run.
-  `interval_ms` of 50–100 ms is recommended for short tests.
-  `guard_samples` adds padding before/after the measured segment for robust capture.
-  `fallback_rails` provides alternatives if the primary rail isn’t found in a given log. It is only used when `rail` is a single string.

### 2) Provide a test tag when calling your benchmark

Pass `_bench_tag` to name the test in `energy_results.csv` and to name each raw power log. The final tag in the CSV will be annotated with the rail(s) used and the number of runs averaged.

```python
tag = f"{model}_I{I}_K{K}_J{J}_B{batch_size}_runs{runs}"

# This will produce a CSV row with a test name like:
# "p3767-0000_I..._runs1(VIN_SYS_5V0, avg of 1 runs)"
# or, for a multi-rail measurement:
# "p3767-0000_I..._runs1(VDD_GPU_SOC+VDD_CPU_CV, avg of 1 runs)"
latency_sec = run_benchmark(op_func, runs, _bench_tag=tag)
```

### 3) Run the GEMM benchmark inside the container

```bash
cd /workspace
python3 gemm.py
```

This produces:

-  `model_benchmarks.csv` — throughput and latency per test
-  `energy_results.csv` — per-test energy and power as rows `[Test, Energy_J, Avg_Power_mW]`
-  `powerlogs/` — raw `tegrastats` logs for each test run

---

## Merging CSVs by Row Order

To create a single CSV with appended energy and power columns (row-aligned by order):

```python
from tegrapower import merge_csvs_by_row_order

merge_csvs_by_row_order(
    benchmark_csv_path="model_benchmarks.csv",
    energy_csv_path="energy_results.csv",
    merged_csv_path="bench_with_energy.csv"
)
```

The merged file will have columns:

-  `model, I, J, K, BATCH_SIZE, throughput_gops, latency_sec, Energy, Avg_Power_mW`

If the energy file has fewer rows than the benchmark file, missing entries are filled with `0.000000` and `0.000`.

---

## Tips for Reliable Energy Capture

Ultra-short tests (microseconds) can be hard to capture. Use these tips:

-  **Decrease sampling interval**:
   - `interval_ms=50` (or 100) for finer sampling.
-  **Add guard samples**:
   - `guard_samples=3` (or higher) to bracket the measurement window with pre/post samples.
-  **Stretch the measured segment**:
   - In `gemm.py`, `run_benchmark` can ensure a minimum measured time (e.g., 0,5–1,0 s) by internally repeating the operation and dividing by repeat count. This preserves per-iteration latency and supplies more samples for integration.
-  **Verify rail presence**:
   - Ensure your chosen `rail`(s) appear in the tegrastats logs (e.g., `VIN_SYS_5V0 4132mW/…`).

---

## Using tegrapower.py Without the Decorator (Manual Mode)

```python
from tegrapower import TegrastatsLogger, summarize_log, estimate_dt_from_interval_ms

# Start logging
logger = TegrastatsLogger("power.log", interval_ms=100)
logger.start()

# ... run your workload ...

# Stop logging
logger.stop()

# Summarize energy (force fixed dt for robustness)
dt = estimate_dt_from_interval_ms(100)
summary = summarize_log("power.log", dt_hint_s=dt, force_fixed_dt=True)

# Example: energy in Joules from an input rail
print(summary["VIN_SYS_5V0"]["energy_J"])

# Example: manually sum energy from multiple rails
gpu_J = summary.get("VDD_GPU_SOC", {}).get("energy_J", 0.0)
cpu_J = summary.get("VDD_CPU_CV", {}).get("energy_J", 0.0)
print(f"Combined GPU+CPU Energy: {gpu_J + cpu_J:.6f} J")
```

---

## tegrapower.py CLI Utilities

Some quick helper commands:

-  Start tegrastats  
   `python3 tegrapower.py start --log power.log --interval-ms 100`

-  Parse a log to stdout  
   `python3 tegrapower.py parse --log power.log --dt-hint 0.1`

-  Export time-series samples to CSV  
   `python3 tegrapower.py samples-csv --log power.log --out samples.csv --dt-hint 0.1`

-  Export per-rail summary to CSV  
   `python3 tegrapower.py summary-csv --log power.log --out summary.csv --dt-hint 0.1`

Optional quick merge from the shell (Python one-liner):

```bash
python3 -c "from tegrapower import merge_csvs_by_row_order as m; m('model_benchmarks.csv','energy_results.csv','bench_with_energy.csv')"
```

---

## FAQ

-  **Why do I sometimes see `0.000000` J?**
   - Not enough samples were captured inside the measured window, or the specified rail(s) were not found in the log. Try:
     - Smaller `interval_ms` (50–100 ms)
     - Larger `guard_samples` (3–5)
     - Stretch the measured segment to ≥ 0,5–1,0 s
     - Confirm the rail name (e.g., `VIN_SYS_5V0`) exists in your `tegrastats` output.

-  **Do I need timestamps in tegrastats?**
   - No. The decorator uses a fixed `dt` per line (`force_fixed_dt=True`) by default to avoid timestamp ambiguity and ensure each sample contributes equally to the energy integral.


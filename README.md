Of course. Based on the new files and changes in your repository, I have updated the `README.md` to reflect the new workflow, features, and file structure.

Here is the updated `README.md`:

# Tegrapower: A Python Toolkit for Energy Measurement on NVIDIA Jetson

## Overview

`tegrapower` is a lightweight Python toolkit for measuring power and energy consumption of code running on NVIDIA Jetson devices. It uses the built-in `tegrastats` utility to capture power data and provides easy-to-use tools to integrate energy consumption directly into your benchmarking workflow.

This repository is designed to streamline the process of running benchmarks, measuring their energy footprint across multiple Jetson devices, and aggregating the results into a single, analysis-ready dataset.

### Key Features

*   **Simple Energy Measurement**: A `@measure_energy_to_csv` decorator to wrap any Python function and automatically measure its energy usage.
*   **Configurable via `.env`**: Easily configure container images, power rails, and measurement settings using a `.env` file.
*   **Multi-Device Result Aggregation**: A script to combine benchmark results from different devices (e.g., AGX Orin, Xavier NX) into a single master CSV.
*   **Makefile-Driven Workflow**: Simple `make` targets to launch the environment, run benchmarks, combine results, and clean up artifacts.
*   **Flexible & Manual Control**: Includes a `TegrastatsLogger` class and parsing utilities for custom, fine-grained power logging.
*   **Example Benchmark**: Comes with a GEMM/BMM benchmark (`gemm.py`) to demonstrate the toolkit's usage.

---

## Repository Structure

```
tegrapower/
├── README.md                 # This file
├── tegrapower.py             # Core library with logger, decorator, and parsers
├── gemm.py                   # Example GEMM/BMM benchmark script
├── makefile                  # Makefile for easy workflow management
├── .env.template             # Template for environment configuration
├── scripts/
│   └── combine_results.py    # Script to merge results from multiple devices
└── results/
    ├── agx/
    │   └── agx_bench_with_energy.csv
    ├── nx/
    │   └── nx_bench_with_energy.csv
    └── orin/
        └── orin_bench_with_energy.csv
```

---

## Requirements

*   An NVIDIA Jetson device.
*   JetPack installed on the host.
*   Docker and the NVIDIA Container Runtime.
*   The `tegrastats` utility (typically found at `/usr/bin/tegrastats` on the host).

---

## Workflow

The workflow is designed to be simple: configure your environment, run the benchmark inside a container, place the results in the correct folder, and then combine them on the host.

### Step 1: Configuration (Host Machine)

First, prepare your environment and configuration.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/iloudaros/tegrapower.git
    cd tegrapower
    ```

2.  **Create `.env` file:**
    Copy the template to create your local configuration file.
    ```bash
    cp .env.template .env
    ```

3.  **Edit `.env`:**
    Open `.env` and customize the variables. The most important one is `CONTAINER`, which should match your Jetson device. Use the `make model` and `make jetpack_version` commands to help identify your system.

    | Jetson Model | Recommended Container |
    | :--- | :--- |
    | Jetson AGX Orin | `nvcr.io/nvidia/l4t-ml:r36.2.0-py3` |
    | Jetson AGX Xavier | `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` |
    | Jetson NX Xavier | `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` |

    You can also configure the power `rail` to measure, the sampling `interval_ms`, and the number of `num_runs` for averaging. See the `.env.template` file for all options.

### Step 2: Run Benchmark (Inside Docker)

1.  **Launch the Docker Container:**
    The Makefile will automatically mount the current directory into `/workspace` inside the container.
    ```bash
    make docker
    ```

2.  **Run the Benchmark Script:**
    Once inside the container, run the example benchmark.
    ```bash
    cd /workspace
    python3 gemm.py
    ```
    This script will:
    *   Run a series of GEMM and BMM operations.
    *   Use the `@measure_energy_to_csv` decorator to log power data for each test.
    *   Generate `model_benchmarks.csv` (latency/throughput) and `energy_results.csv`.
    *   Automatically merge them into a final `bench_with_energy.csv` file.

### Step 3: Aggregate Results (Host Machine)

After running the benchmark on one or more devices, you can combine the output files.

1.  **Organize Result Files:**
    On your host machine, create a subdirectory inside `results/` named after the device you tested (e.g., `agx`, `nx`, `orin-agx`). Copy the `bench_with_energy.csv` file you generated into this new directory. For example, for an AGX Orin:
    ```bash
    mkdir -p results/agx-orin
    cp bench_with_energy.csv results/agx-orin/
    ```
    Repeat this process for each device you benchmark. The `results` directory already contains examples for `agx`, `nx`, and `orin`.

2.  **Combine All Results:**
    Run the `combine` target from the Makefile on your host.
    ```bash
    make combine
    ```
    This will execute `scripts/combine_results.py`, which scans all subdirectories in `results/`, and creates a single `results/combined_results.csv` file with an added `device` column.

---

## Using the Makefile

The `Makefile` provides several convenient targets to manage the workflow from the host machine.

*   `make docker`: Launches the pre-configured NVIDIA container with your workspace mounted.
*   `make combine`: Scans the `results` directory and combines all `*.csv` files into a single master file.
*   `make clean`: Removes all generated files (`*.csv`, `powerlogs/`, `__pycache__/`).
*   `make model`: Prints the model of your Jetson device.
*   `make update`: Pulls the latest version of the repository from git.
*   `make jetpack_version`: Shows JetPack and L4T release details.

---

## Customizing Your Benchmark

To measure energy for your own code, you can use the `@measure_energy_to_csv` decorator provided in `tegrapower.py`.

### Decorator Usage

The decorator reads its configuration from environment variables, which you can set in the `.env` file.

In your benchmark script, first load the environment variables, then apply the decorator to your function.

```python
from tegrapower import measure_energy_to_csv, load_dot_env, extract_env_var

# Load .env file at the start of your script
load_dot_env()

@measure_energy_to_csv(
    # These values are pulled from environment variables
    rail=extract_env_var("rail", list, default="VIN_SYS_5V0"),
    interval_ms=extract_env_var("interval_ms", int, default=50),
    log_dir=extract_env_var("log_dir", str, default="powerlogs"),
    num_runs=extract_env_var("num_runs", int, default=5), # Averages energy over 5 runs
    energy_csv_path=extract_env_var("energy_csv_path", str, default="energy_results.csv"),
    # ... other parameters
)
def my_benchmark_function(args):
    # Your code to be measured goes here
    # ...
    return result

# When calling, pass a '_bench_tag' to identify the test in the output CSV
my_benchmark_function(my_args, _bench_tag="My_Test_1")

```

**Key parameters controlled by `.env`:**
*   `rail`: The power rail(s) to measure. Can be a single rail (`"VIN_SYS_5V0"`) or a comma-separated list (`"VDD_GPU_SOC,VDD_CPU_CV"`), which will be summed.
*   `num_runs`: The decorator will execute the decorated function this many times and average the energy results into a single output row.
*   `interval_ms`: The sampling rate for `tegrastats`. 50-100ms is good for short tests.

> [!TIP]
>**Note on Rails:** To find the correct rail names for your device, run `tegrastats` in your terminal and inspect the output. We recommend the following as starting points:
>
> | Jetson Model | Common Rail Names |
> | :--- | :--- |
> | Jetson AGX Orin | `VDD_GPU_SOC`, `VDD_CPU_CV`, `VIN_SYS_5V0` |
> | Jetson AGX Xavier | `GPU`, `CPU`, `SOC`, `CV`, `VDDRQ`, `SYS5V` |
> | Jetson NX Xavier | `VDD_IN` |

---

## Scripts & CLI

### `tegrapower.py`
Beyond the decorator, `tegrapower.py` can be used as a command-line tool for manual logging and parsing.

*   **Start logging**:
    `python3 tegrapower.py start --log power.log --interval-ms 100`
*   **Parse a log and print summary**:
    `python3 tegrapower.py parse --log power.log`
*   **Export time-series data to CSV**:
    `python3 tegrapower.py samples-csv --log power.log --out samples.csv`
*   **Export summary to CSV**:
    `python3 tegrapower.py summary-csv --log power.log --out summary.csv`

### `scripts/combine_results.py`
This script aggregates data. It's typically run via `make combine`, but can also be called directly:
```bash
python3 scripts/combine_results.py --results-dir results --output-file results/all_benchmarks.csv
```

---

## License
This project is licensed under the **CC0 1.0 Universal** public domain dedication. See the [LICENSE](LICENSE) file for details. You are free to use, modify, and distribute this work for any purpose.


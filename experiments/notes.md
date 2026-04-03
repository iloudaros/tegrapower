# Notes about the experiments 

## Useful information about Power Modes and Clock Frequencies
You can find more information about Clock Frequency and Power Modes in the [NVIDIA Jetson Linux Developer Guide](https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3275/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_jetson_xavier.html#wwpID0E04T0HA.)


## Running the experiments
We provide two primary experiment scripts:
	1.	`⁠jetson_nvpmodel.py`: Sweeps across NVIDIA Jetson power modes (⁠nvpmodel).	
    2.	`⁠jetson_freq.py`: Sweeps across specific GPU clock frequencies via ⁠sysfs.
Both scripts automatically discover the capabilities of your specific Jetson board, execute the benchmark, clean up intermediate artifacts to prevent data corruption, and output a unique CSV file for every hardware configuration tested.

### Prerequisites & Docker Setup

Because these experiments manipulate low-level hardware states (CPU cores, clock limits, and GPU frequencies), they require elevated privileges.

*   **Running locally**: You must run the scripts using `sudo`.
*   **Running in Docker**: You must grant the container access to the host's hardware control mechanisms. Ensure your `makefile` or `docker run` command includes the following:
    *   The `--privileged` flag (to allow writing to `/sys/devices/...`).
    *   Volume mount for the `nvpmodel` binary: `-v /usr/sbin/nvpmodel:/usr/sbin/nvpmodel`
    *   Volume mount for the power configs: `-v /etc/nvpmodel.conf:/etc/nvpmodel.conf`




### Experiment 1: Power Modes (`jetson_nvpmodel.py`)

This script tests your workload across different predefined NVIDIA power profiles. It reads `/etc/nvpmodel.conf` to discover valid modes for your device, applies them sequentially, and benchmarks each one.

#### Usage

Run the script from the root of the repository:

```bash
sudo python3 experiments/jetson_nvpmodel.py [OPTIONS]
```

#### Command-Line Arguments

*   `--modes`: Space-separated list of specific power mode IDs to test. If omitted, the script tests all available modes on the device.

#### Examples

*   **Test all available power modes:**
    ```bash
    sudo python3 experiments/jetson_nvpmodel.py
    ```
*   **Test specific modes (e.g., Mode 0, 2, and 4):**
    ```bash
    sudo python3 experiments/jetson_nvpmodel.py --modes 0 2 4
    ```

#### Outputs
For each mode tested, a distinct CSV file is generated in the parent directory:
`bench_with_energy_mode_<MODE_ID>.csv`

### Experiment 2: Clock Frequencies (`jetson_freq.py`)

This script provides granular control over the GPU performance by sweeping across individual clock frequencies. It targets the specific `sysfs` devfreq path for your Jetson model (Nano, NX, AGX, or Orin) and iterates through the `available_frequencies` list.

#### Usage

Run the script from the root of the repository:

```bash
sudo python3 experiments/jetson_freq.py [OPTIONS]
```

#### Command-Line Arguments

You can specify exact frequencies or provide a bounded range. Frequencies must be provided in **Hertz (Hz)**.

*   `--freqs`: Space-separated list of specific GPU frequencies to test.
*   `--min-freq`: The minimum GPU frequency to test (inclusive).
*   `--max-freq`: The maximum GPU frequency to test (inclusive).

*If no arguments are provided, the script tests every supported GPU frequency.*

#### Examples

*   **Test all available GPU frequencies:**
    ```bash
    sudo python3 experiments/jetson_freq.py
    ```
*   **Test all frequencies above 500 MHz (500,000,000 Hz):**
    ```bash
    sudo python3 experiments/jetson_freq.py --min-freq 500000000
    ```
*   **Test a frequency range (e.g., 300 MHz to 800 MHz):**
    ```bash
    sudo python3 experiments/jetson_freq.py --min-freq 300000000 --max-freq 800000000
    ```
*   **Test specific discrete frequencies:**
    ```bash
    sudo python3 experiments/jetson_freq.py --freqs 114750000 510000000 1300500000
    ```

### Outputs
For each frequency tested, a distinct CSV file is generated in the parent directory:
`bench_with_energy_freq_<FREQUENCY_IN_HZ>.csv`
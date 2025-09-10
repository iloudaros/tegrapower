import os
import re
import shlex
import signal
import subprocess
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from functools import wraps
import time


class TegrastatsLogger:
    """
    Manages a background tegrastats process and streams its output to a file.

    Typical usage:
        logger = TegrastatsLogger("power.log", interval_ms=200)
        logger.start()
        # ... run workload ...
        logger.stop()

    Notes:
    - tegrastats is expected to be in the system's PATH.
    - interval_ms controls the sampling frequency.
    - Uses a process group to cleanly send signals to the child process.
    """

    def __init__(self, log_path: str, interval_ms: int = 200, tegrastats_path: str = "tegrastats"):
        self.log_path = os.path.abspath(log_path)
        self.interval_ms = int(interval_ms)
        self.tegrastats_path = tegrastats_path
        self._proc: Optional[subprocess.Popen] = None

    def start(self, append: bool = True) -> None:
        """
        Starts the tegrastats process and directs its output to self.log_path.

        Args:
            append (bool): 
                - True: Appends to the existing log file (if any).
                - False: Overwrites the log file.
        """
        # Prevent starting if a process is already running
        if self._proc is not None and self._proc.poll() is None:
            raise RuntimeError("tegrastats is already running")

        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        # Set file mode to append-binary or write-binary
        mode = "ab" if append else "wb"
        f = open(self.log_path, mode)

        # Construct the command and spawn the process
        cmd = f"{shlex.quote(self.tegrastats_path)} --interval {self.interval_ms}"
        try:
            # preexec_fn=os.setsid creates a new process group, allowing us to
            # send signals to the entire group, ensuring clean termination.
            self._proc = subprocess.Popen(
                shlex.split(cmd),
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
        except FileNotFoundError as e:
            f.close()
            raise RuntimeError("tegrastats executable not found in PATH.") from e
        except Exception:
            f.close()
            raise

    def is_running(self) -> bool:
        """Returns True if the tegrastats process is currently active."""
        return self._proc is not None and self._proc.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stops the tegrastats process gracefully.

        Sends SIGTERM first, and if the process doesn't exit within the timeout,
        sends SIGKILL to force termination.

        Args:
            timeout (float): Seconds to wait after SIGTERM before sending SIGKILL.
        """
        if not self.is_running():
            return
        
        # Send SIGTERM to the entire process group
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            # Process might have already terminated
            pass

        # Wait for the process to terminate
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # If it doesn't terminate, force kill it with SIGKILL
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                # Wait again to ensure the process is reaped
                self._proc.wait(timeout=1.0)
        finally:
            self._proc = None


# ---------- Parsing tailored to Jetson-style tegrastats lines ----------

# Regex to capture power rail names and their instantaneous power values (mW).
# e.g., "VDD_GPU_SOC 2405mW/2405mW" -> captures ("VDD_GPU_SOC", "2405")
RAIL_PAIR_RE = re.compile(
    r"\b([A-Za-z0-9_]+)\s+([0-9]{1,8})\s*mW(?:\s*\/\s*[0-9]{1,8}\s*mW)?\b",
    re.IGNORECASE
)

# Regex to find optional wall-clock timestamps (e.g., "09-02-2025 10:30:00")
WALLCLOCK_RE = re.compile(r"\b(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2})\b")
# Regex to find optional monotonic timestamps in brackets (e.g., "[ 1234.56 ]")
BRACKET_SEC_RE = re.compile(r"\[\s*([0-9]+(?:\.[0-9]+)?)\s*\]")


def parse_tegrastats_line(line: str) -> Dict[str, float]:
    """
    Extracts all rail -> instantaneous power (mW) key-value pairs from a single log line.
    """
    rails: Dict[str, float] = {}
    for name, val in RAIL_PAIR_RE.findall(line):
        rails[name] = float(val)
    return rails


def parse_times_for_line(line: str) -> Optional[float]:
    """
    Parses an optional timestamp from a line, returning a POSIX timestamp (float seconds).
    Returns None if no parsable timestamp is found.
    """
    # Check for "MM-DD-YYYY HH:MM:SS" format
    m = WALLCLOCK_RE.search(line)
    if m:
        ts = m.group(1)
        try:
            dt = datetime.strptime(ts, "%m-%d-%Y %H:%M:%S")
            return dt.timestamp()
        except ValueError:
            return None

    # Check for "[ 1234.56 ]" format
    b = BRACKET_SEC_RE.search(line)
    if b:
        try:
            return float(b.group(1))
        except ValueError:
            return None

    return None


def parse_log(filepath: str) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Parses a tegrastats log file.

    Returns:
        - A list of relative timestamps in seconds (if available).
        - A list of power sample dictionaries (rail -> power_mW).

    If timestamps are missing or inconsistent, the times list will be empty.
    """
    times_abs: List[Optional[float]] = []  # Absolute POSIX timestamps
    samples: List[Dict[str, float]] = []   # Power samples

    # Read all lines from the log file, decoding with error handling
    raw_lines: List[str] = []
    with open(filepath, "rb") as f:
        for b in f:
            line = b.decode("utf-8", errors="ignore").strip()
            if line:
                raw_lines.append(line)

    if not raw_lines:
        return [], []

    # Parse each line for timestamps and power rail data
    for line in raw_lines:
        times_abs.append(parse_times_for_line(line))
        rails = parse_tegrastats_line(line)
        if rails:
            samples.append(rails)

    if not samples:
        return [], []

    # Convert absolute times to relative seconds, if timestamps per raw line exist
    times_s: List[float] = []
    if len(times_abs) == len(raw_lines) and all(t is not None for t in times_abs):
        t0 = times_abs[0]  # type: ignore
        times_s = [float(t - t0) for t in times_abs]  # type: ignore

        # If counts mismatch (e.g., some lines lacked rails), fall back to empty times
        if len(times_s) != len(samples):
            times_s = []

    return times_s, samples


# ---------- Energy Integration ----------

def compute_energy(
    times_s: List[float],
    samples: List[Dict[str, float]],
    dt_hint_s: Optional[float] = None,
    force_fixed_dt: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Integrates per-rail energy from a series of power samples using the trapezoidal rule.

    Args:
        times_s: Relative timestamps (seconds) for each sample. Can be empty.
        samples: List of power samples, where each sample is a dict of {rail: power_mW}.
        dt_hint_s: Sampling interval (seconds) to use if timestamps are missing or forced.
        force_fixed_dt: If True, ignores `times_s` and uses a constant `dt_hint_s`.

    Returns:
        A dictionary mapping each rail to its average power, total energy in Joules,
        and total energy in Watt-hours.
    """
    if not samples:
        return {}

    # Default time delta is 0.2s if not provided
    base_dt = dt_hint_s if dt_hint_s is not None else 0.2

    # Determine the time delta (dt) for each sample interval
    dts: List[float]
    if (not force_fixed_dt) and times_s and len(times_s) >= len(samples):
        # Use actual time differences if available and not forced
        dts = [times_s[i + 1] - times_s[i] for i in range(len(samples) - 1)]
        dts = [max(0.0, dt) for dt in dts] # Ensure no negative time travel
        # Append a final dt, using the median dt or base_dt as an estimate
        dts.append(dts[len(dts)//2] if dts else base_dt)
    else:
        # Use a fixed time delta for all samples
        dts = [base_dt] * len(samples)

    # Collect all unique rail names from all samples
    all_rails = sorted({k for s in samples for k in s.keys()})
    results: Dict[str, Dict[str, float]] = {}

    # Calculate energy for each rail
    for rail in all_rails:
        energy_mJ = 0.0
        total_power = 0.0
        for i, sample in enumerate(samples):
            power_mW = float(sample.get(rail, 0.0))
            dt = dts[i] if i < len(dts) else dts[-1]  # Use last dt if list is short
            # Energy (mJ) = Power (mW) * time (s)
            energy_mJ += power_mW * dt
            total_power += power_mW
        
        avg_power_mW = total_power / max(1, len(samples))
        energy_J = energy_mJ / 1000.0
        energy_Wh = energy_J / 3600.0
        
        results[rail] = {
            "avg_power_mW": avg_power_mW,
            "energy_J": energy_J,
            "energy_Wh": energy_Wh,
        }

    # Optional: Calculate a sum of all rails that are not considered primary input sources
    # This helps avoid double-counting energy.
    if all_rails:
        sum_energy_J = 0.0
        sum_power_mW = 0.0
        input_rails = {"VIN_SYS_5V0", "POM_5V_IN", "TOTAL"}
        for rail in all_rails:
            if rail not in input_rails:
                sum_energy_J += results[rail]["energy_J"]
                sum_power_mW += results[rail]["avg_power_mW"]
        
        if sum_energy_J > 0:
            results["SUM_NO_INPUT"] = {
                "avg_power_mW": sum_power_mW,
                "energy_J": sum_energy_J,
                "energy_Wh": sum_energy_J / 3600.0,
            }

    return results


def summarize_log(filepath: str, dt_hint_s: Optional[float] = None, force_fixed_dt: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to parse a log file and compute the energy summary.
    `force_fixed_dt=True` is useful for ensuring consistent energy calculation across samples.
    """
    times_s, samples = parse_log(filepath)
    return compute_energy(times_s, samples, dt_hint_s=dt_hint_s, force_fixed_dt=force_fixed_dt)


def estimate_dt_from_interval_ms(interval_ms: int) -> float:
    """Converts tegrastats interval (ms) to seconds, with a minimum of 1 ms."""
    return max(0.001, float(interval_ms) / 1000.0)


# ---------- CSV Export Utilities ----------

def export_samples_csv(
    filepath_in: str,
    filepath_out: str,
    dt_hint_s: Optional[float] = None,
    rails_filter: Optional[List[str]] = None
) -> None:
    """
    Writes time-series power data to a CSV file (time_s, rail1_mW, rail2_mW, ...).

    If timestamps are missing, a time axis is synthesized using `dt_hint_s`.
    `rails_filter` can be used to specify which rails to include and in what order.
    """
    times_s, samples = parse_log(filepath_in)
    if not samples:
        # Create an empty CSV with just a header if there are no samples
        with open(filepath_out, "w", newline="") as f:
            csv.writer(f).writerow(["time_s"])
        return

    # Determine which rails to include in the CSV
    all_rails = sorted({k for s in samples for k in s.keys()})
    if rails_filter:
        # Use the user-provided filter
        rails = [r for r in rails_filter if r in all_rails]
    else:
        # Default ordering: prioritize input/total rails, then the rest alphabetically
        priority = [r for r in ("POM_5V_IN", "VIN_SYS_5V0", "TOTAL") if r in all_rails]
        rest = sorted([r for r in all_rails if r not in priority])
        rails = priority + rest

    # Synthesize a time axis if it's missing or inconsistent
    if not times_s or len(times_s) < len(samples):
        base_dt = dt_hint_s if dt_hint_s is not None else 0.2
        times_s = [i * base_dt for i in range(len(samples))]
    elif len(times_s) > len(samples):
        times_s = times_s[:len(samples)]

    # Write data to CSV
    with open(filepath_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s"] + rails)  # Header row
        for i, s in enumerate(samples):
            # Write one row per sample
            row = [f"{times_s[i]:.6f}"] + [f"{float(s.get(r, 0.0)):.3f}" for r in rails]
            writer.writerow(row)


def export_summary_csv(
    filepath_in: str,
    filepath_out: str,
    dt_hint_s: Optional[float] = None,
    include_sum_no_input: bool = True
) -> None:
    """
    Writes a per-rail energy summary to a CSV file.
    Columns: rail, avg_power_mW, energy_J, energy_Wh
    """
    summary = summarize_log(filepath_in, dt_hint_s=dt_hint_s, force_fixed_dt=True)
    with open(filepath_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rail", "avg_power_mW", "energy_J", "energy_Wh"])  # Header
        for rail, vals in summary.items():
            if not include_sum_no_input and rail == "SUM_NO_INPUT":
                continue
            w.writerow([
                rail,
                f"{vals['avg_power_mW']:.3f}",
                f"{vals['energy_J']:.6f}",
                f"{vals['energy_Wh']:.9f}",
            ])


# ---------- Decorator for Measuring Energy of a Function ----------

def measure_energy_to_csv(
    rail: Union[str, List[str]] = "VIN_SYS_5V0",
    interval_ms: int = 200,
    log_dir: str = "powerlogs",
    num_runs: int = 1,
    guard_samples: int = 2,
    energy_csv_path: str = "energy_results.csv",
    append: bool = True,
    also_write_log_file: bool = True,
    fallback_rails: Optional[List[str]] = None
):
    """
    A decorator to measure the energy consumption of a function call and log the results.

    It wraps the function, starts tegrastats logging before execution, stops it after,
    calculates the energy consumed, and appends the result to a CSV file.

    Args:
        rail (Union[str, List[str]]): The primary power rail(s) to measure. If a string, it's the main rail.
                                       If a list of strings, the energy/power of those rails are summed up.
        interval_ms (int): Sampling interval for tegrastats in milliseconds.
        log_dir (str): Base directory to store power logs.
        num_runs (int): Number of times to execute the function and average the results.
        guard_samples (int): Number of sampling intervals to wait before and after the function call.
        energy_csv_path (str): Path to the output CSV for energy results.
        append (bool): If True, append to the CSV; otherwise, create a new one.
        also_write_log_file (bool): If True, perform logging. If False, just run the function.
        fallback_rails (Optional[List[str]]): List of alternative rails to try if the primary is not found (only used if `rail` is a string).

    Usage:
        @measure_energy_to_csv(rail=["VDD_GPU_SOC", "VDD_CPU_CV"])
        def run_benchmark(op_func, runs):
            ...

        # At call site:
        run_benchmark(op_func, runs, _bench_tag="Model_I..._K..._J..._B..._runs...")

    Behavior:
      - For each of the `num_runs`:
        - Creates a numbered subfolder in `log_dir` (e.g., `powerlogs/1`, `powerlogs/2`).
        - Starts tegrastats, logging to a file within the subfolder.
        - Executes the decorated function.
        - Stops tegrastats.
      - After all runs, it calculates the average energy and power over all successful runs.
      - If `rail` was a list, the values are summed for each run before averaging.
      - Appends a single row with the averaged results to `energy_csv_path`.
      - The 'Test' column in the CSV is annotated to show the number of runs averaged and the rail(s) used.
    """
    os.makedirs(log_dir, exist_ok=True)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract a custom tag for the test run from keyword arguments
            tag = kwargs.pop("_bench_tag", "bench")
            dt_hint = estimate_dt_from_interval_ms(interval_ms)

            # Handle CSV header creation/deletion
            if not append and os.path.exists(energy_csv_path):
                os.remove(energy_csv_path)
            need_header = not os.path.exists(energy_csv_path)

            # If logging is disabled, just run the function and write dummy results
            if not also_write_log_file:
                result = None
                for _ in range(num_runs):
                    result = func(*args, **kwargs)
                with open(energy_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    if need_header:
                        writer.writerow(["Test", "Energy_J", "Avg_Power_mW"])
                    writer.writerow([tag, "0.000000", "0.000"])
                return result

            run_energies: List[float] = []
            run_powers: List[float] = []
            rail_display_name = ""
            any_samples_found = False
            result = None

            # Execute the function `num_runs` times
            for i in range(num_runs):
                run_log_dir = os.path.join(log_dir, str(i + 1))
                os.makedirs(run_log_dir, exist_ok=True)
                log_path = os.path.join(run_log_dir, f"{tag}_{int(time.time())}.log")
                logger = TegrastatsLogger(log_path, interval_ms=interval_ms)

                try:
                    logger.start(append=False)
                    # "Guard time" to ensure logging starts before the workload
                    if guard_samples > 0:
                        time.sleep(dt_hint * guard_samples)
                    
                    # Execute the decorated function
                    result = func(*args, **kwargs)
                    
                    # "Guard time" to ensure the workload finishes before logging stops
                    if guard_samples > 0:
                        time.sleep(dt_hint * guard_samples)
                finally:
                    # Ensure the logger is always stopped
                    logger.stop()

                # Check if the log file contains any data
                line_count = 0
                if os.path.exists(log_path):
                    try:
                        with open(log_path, "r", encoding="utf-8", errors="ignore") as lf:
                            line_count = sum(1 for _ in lf)
                    except Exception:
                        pass
                
                if line_count == 0:
                    continue  # Skip this run if the log is empty

                any_samples_found = True
                # Analyze the log file to get the energy summary
                summary = summarize_log(log_path, dt_hint_s=dt_hint, force_fixed_dt=True)
                
                run_energy_J = 0.0
                run_avg_power_mW = 0.0
                rails_used_this_run = []
                
                # If `rail` is a list, sum the energy and power from all specified rails
                if isinstance(rail, list):
                    for r_name in rail:
                        if r_name in summary:
                            vals = summary[r_name]
                            run_energy_J += float(vals.get("energy_J", 0.0))
                            run_avg_power_mW += float(vals.get("avg_power_mW", 0.0))
                            rails_used_this_run.append(r_name)
                    if rails_used_this_run:
                        rail_display_name = "+".join(rails_used_this_run)
                # If `rail` is a single string, find its data
                elif isinstance(rail, str):
                    chosen_rail_name = rail
                    vals = summary.get(chosen_rail_name)

                    # If primary rail is not found, try the fallback rails
                    if not vals and fallback_rails:
                        for fr in fallback_rails:
                            if fr in summary:
                                chosen_rail_name = fr
                                vals = summary.get(fr)
                                break
                    
                    if vals:
                        run_energy_J = float(vals.get("energy_J", 0.0))
                        run_avg_power_mW = float(vals.get("avg_power_mW", 0.0))
                        rails_used_this_run.append(chosen_rail_name)
                        rail_display_name = chosen_rail_name
                
                # If we successfully found data for the requested rail(s), store it
                if rails_used_this_run:
                    run_energies.append(run_energy_J)
                    run_powers.append(run_avg_power_mW)

            # After all runs, calculate the final average energy and power
            final_energy_J = 0.0
            final_avg_power_mW = 0.0
            tag_to_write = tag

            if run_energies:
                # Average the results from all successful runs
                avg_energy = sum(run_energies) / len(run_energies)
                avg_power = sum(run_powers) / len(run_powers)
                final_energy_J = avg_energy
                final_avg_power_mW = avg_power
                tag_to_write = f"{tag}({rail_display_name}, avg of {len(run_energies)} runs)"
            elif any_samples_found:
                # Samples were found, but not for the specified rail(s)
                tag_to_write = f"{tag}(NO_RAIL)"
            else:
                # No samples were collected at all
                tag_to_write = f"{tag}(NO_SAMPLES)"

            # Write the final averaged results to the CSV
            with open(energy_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if need_header:
                    writer.writerow(["Test", "Energy_J", "Avg_Power_mW"])
                writer.writerow([tag_to_write, f"{final_energy_J:.6f}", f"{final_avg_power_mW:.3f}"])

            return result
        return wrapper
    return decorator


def merge_csvs_by_row_order(
    benchmark_csv_path: str,
    energy_csv_path: str,
    merged_csv_path: str
) -> None:
    """
    Merges two CSV files by appending columns from the second file to the first.
    It assumes a 1-to-1 correspondence between the rows of the two files.

    If the energy file has fewer rows than the benchmark file, missing energy
    values are filled with 0.
    """
    # Read all rows from the benchmark CSV
    with open(benchmark_csv_path, "r", encoding="utf-8", errors="ignore") as fb:
        bench_reader = csv.reader(fb)
        bench_rows = list(bench_reader)

    if not bench_rows:
        raise ValueError("Benchmark CSV is empty")

    # Read all rows from the energy CSV
    with open(energy_csv_path, "r", encoding="utf-8", errors="ignore") as fe:
        energy_reader = csv.reader(fe)
        energy_rows = list(energy_reader)

    # If energy file is empty, just add zero columns to the benchmark data
    if not energy_rows:
        header = bench_rows[0] + ["Energy_J", "Avg_Power_mW"]
        with open(merged_csv_path, "w", newline="") as fo:
            writer = csv.writer(fo)
            writer.writerow(header)
            for r in bench_rows[1:]:
                writer.writerow(r + ["0.000000", "0.000"])
        return

    bench_header = bench_rows[0]
    energy_header = energy_rows[0]

    # Find the indices of the energy and power columns in the energy CSV
    try:
        energy_idx = energy_header.index("Energy_J")
        power_idx = energy_header.index("Avg_Power_mW")
    except ValueError:
        raise ValueError("Energy CSV must have columns 'Energy_J' and 'Avg_Power_mW'")

    # Prepare the output rows with the new merged header
    out_header = bench_header + ["Energy_J", "Avg_Power_mW"]
    out_rows = [out_header]

    bench_data = bench_rows[1:]
    energy_data = energy_rows[1:]

    # Iterate through benchmark rows and append corresponding energy data
    for i, brow in enumerate(bench_data):
        if i < len(energy_data):
            erow = energy_data[i]
            # Defensively get values, defaulting to 0 if columns are missing
            energy_val = erow[energy_idx] if len(erow) > energy_idx else "0.000000"
            power_val = erow[power_idx] if len(erow) > power_idx else "0.000"
        else:
            # If no more energy rows, pad with zeros
            energy_val = "0.000000"
            power_val = "0.000"
        out_rows.append(brow + [energy_val, power_val])

    # Write the merged data to the output CSV
    with open(merged_csv_path, "w", newline="") as fo:
        writer = csv.writer(fo)
        writer.writerows(out_rows)


# ---------- Command-Line Interface ----------
if __name__ == "__main__":
    # This block provides a simple CLI for using the script's functions directly.
    import argparse
    parser = argparse.ArgumentParser(description="Tegrastats power logger, parser, and energy calculator")
    subparsers = parser.add_subparsers(dest="cmd", help="Available commands")

    # Sub-command to start the tegrastats logger
    p_start = subparsers.add_parser("start", help="Start tegrastats logging to a file.")
    p_start.add_argument("--log", required=True, help="Path to the output log file.")
    p_start.add_argument("--interval-ms", type=int, default=200, help="Logging interval in milliseconds.")
    p_start.add_argument("--overwrite", action="store_true", help="Overwrite the log file if it exists.")

    # Sub-command to parse a log and print a summary
    p_parse = subparsers.add_parser("parse", help="Parse a log file and print an energy summary to the console.")
    p_parse.add_argument("--log", required=True, help="Path to the log file to parse.")
    p_parse.add_argument("--dt-hint", type=float, default=None, help="Time delta hint in seconds for energy calculation.")

    # Sub-command to export time-series samples to CSV
    p_samples = subparsers.add_parser("samples-csv", help="Export time-series power samples to a CSV file.")
    p_samples.add_argument("--log", required=True, help="Path to the input log file.")
    p_samples.add_argument("--out", required=True, help="Path to the output CSV file.")
    p_samples.add_argument("--dt-hint", type=float, default=None, help="Time delta hint in seconds if timestamps are missing.")
    p_samples.add_argument("--rails", nargs="*", default=None, help="Optional list of specific rails to include.")

    # Sub-command to export a summary CSV
    p_summary = subparsers.add_parser("summary-csv", help="Export a per-rail energy summary to a CSV file.")
    p_summary.add_argument("--log", required=True, help="Path to the input log file.")
    p_summary.add_argument("--out", required=True, help="Path to the output summary CSV file.")
    p_summary.add_argument("--dt-hint", type=float, default=None, help="Time delta hint in seconds.")
    p_summary.add_argument("--no-sum", action="store_true", help="Do not include the 'SUM_NO_INPUT' rail in the output.")
    
    # Sub-command to merge benchmark and energy CSVs
    p_merge = subparsers.add_parser("merge-csv", help="Merge a benchmark CSV with an energy CSV.")
    p_merge.add_argument("--bench-csv", required=True, help="Path to the benchmark results CSV.")
    p_merge.add_argument("--energy-csv", required=True, help="Path to the energy results CSV.")
    p_merge.add_argument("--out", required=True, help="Path for the final merged CSV file.")

    args = parser.parse_args()

    # Execute the chosen command
    if args.cmd == "start":
        logger = TegrastatsLogger(args.log, interval_ms=args.interval_ms)
        logger.start(append=not args.overwrite)
        print(f"Started tegrastats -> {logger.log_path} at {args.interval_ms} ms")
    elif args.cmd == "parse":
        dt = args.dt_hint
        if dt is None:
            # Try to infer from interval, if not specified.
            # This part is a placeholder, as interval isn't passed to parse.
            # User should specify --dt-hint for accuracy.
            pass
        summary = summarize_log(args.log, dt_hint_s=dt, force_fixed_dt=bool(dt))
        print("Rail          | Avg Power (mW) | Energy (J) | Energy (Wh)")
        print("----------------------------------------------------------")
        for r, v in summary.items():
            print(f"{r:<13} | {v['avg_power_mW']:<14.2f} | {v['energy_J']:<10.6f} | {v['energy_Wh']:<11.9f}")
    elif args.cmd == "samples-csv":
        export_samples_csv(args.log, args.out, dt_hint_s=args.dt_hint, rails_filter=args.rails)
        print(f"Wrote samples CSV -> {args.out}")
    elif args.cmd == "summary-csv":
        export_summary_csv(args.log, args.out, dt_hint_s=args.dt_hint, include_sum_no_input=not args.no_sum)
        print(f"Wrote summary CSV -> {args.out}")
    elif args.cmd == "merge-csv":
        merge_csvs_by_row_order(args.bench_csv, args.energy_csv, args.out)
        print(f"Merged CSVs into -> {args.out}")
    else:
        parser.print_help()



# # ---------- .env File Loader ----------
def load_dot_env(dotenv_path=".env"):
    """
    Manually loads environment variables from a .env file.
    """
    if not os.path.exists(dotenv_path):
        print(f"Warning: '{dotenv_path}' file not found. Skipping.")
        return False # Indicate that the file was not found

    try:
        with open(dotenv_path) as f:
            for line in f:
                # Ignore comments and empty lines
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Use a regular expression to split by the first '='
                # This handles cases where the value might contain '='
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove surrounding quotes (single or double) from the value
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]

                    # Set the environment variable for the current process
                    os.environ[key] = value
        return True # Indicate that the file was successfully loaded
    except IOError as e:
        print(f"Error reading '{dotenv_path}': {e}")

def extract_env_var(name, type, default=None) -> Any:
    """
    Retrieves and converts an environment variable to the specified type. If the variable is not found, returns the default value if provided, otherwise raises an error.
    """
    value = os.getenv(name)

    if value is None and default is not None:
        return default
    if value is None:
        raise ValueError(f"Environment variable '{name}' not found and default not set.")
    
    processed_value: Any

    try:
        if type == int:
            processed_value = int(value)
        elif type == bool:
            # Handles 'True', 'true', '1' as True, everything else as False
            processed_value = value.lower() in ('true', '1')
        elif type == list:
            # Handles comma-separated strings, strips whitespace from each item
            processed_value = [item.strip() for item in value.split(',')]
        elif type == str:
            # Special handling for 'rail' which has a trailing comma
            if name == 'rail':
                processed_value = value.strip(',')
            else:
                processed_value = value
        else:
            processed_value = value

        return processed_value
    except ValueError as e:
        raise ValueError(f"Error converting environment variable '{name}' to {type}: {e}")
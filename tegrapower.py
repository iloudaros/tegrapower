import os
import re
import shlex
import signal
import subprocess
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable, Any
from functools import wraps
import time


class TegrastatsLogger:
    """
    Manage a background tegrastats process and stream its output to a file.

    Typical usage:
        logger = TegrastatsLogger("power.log", interval_ms=200)
        logger.start()
        # ... run workload ...
        logger.stop()

    Notes:
    - tegrastats is expected to be in PATH.
    - interval_ms controls the sampling cadence.
    - Uses a process group so we can send signals cleanly to the child.
    """

    def __init__(self, log_path: str, interval_ms: int = 200, tegrastats_path: str = "tegrastats"):
        self.log_path = os.path.abspath(log_path)
        self.interval_ms = int(interval_ms)
        self.tegrastats_path = tegrastats_path
        self._proc: Optional[subprocess.Popen] = None

    def start(self, append: bool = True) -> None:
        """
        Start tegrastats and write its stdout to self.log_path.

        append:
            - True: append to existing log (binary mode)
            - False: truncate and start fresh
        """
        if self._proc is not None and self._proc.poll() is None:
            raise RuntimeError("tegrastats is already running")

        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        mode = "ab" if append else "wb"
        f = open(self.log_path, mode)

        # Build and spawn the command
        cmd = f"{shlex.quote(self.tegrastats_path)} --interval {self.interval_ms}"
        try:
            self._proc = subprocess.Popen(
                shlex.split(cmd),
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # create a process group for clean termination
            )
        except FileNotFoundError as e:
            f.close()
            raise RuntimeError("tegrastats executable not found in PATH.") from e
        except Exception:
            f.close()
            raise

    def is_running(self) -> bool:
        """Return True if the tegrastats process is alive."""
        return self._proc is not None and self._proc.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop tegrastats gracefully, and force kill if it doesn't exit in time.

        timeout:
            Seconds to wait after SIGTERM before sending SIGKILL.
        """
        if not self.is_running():
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                self._proc.wait(timeout=1.0)
        finally:
            self._proc = None


# ---------- Parsing tailored to Jetson-style lines ----------

# We expect tokens like:
#   "VDD_GPU_SOC 2405mW/2405mW VDD_CPU_CV 1603mW/1603mW VIN_SYS_5V0 3931mW/3931mW"
# This regex captures NAME and the instantaneous value (the first number before mW).
RAIL_PAIR_RE = re.compile(
    r"\b([A-Za-z0-9_]+)\s+([0-9]{1,8})\s*mW(?:\s*\/\s*[0-9]{1,8}\s*mW)?\b",
    re.IGNORECASE
)

# Optional timestamps supported:
# - "MM-DD-YYYY HH:MM:SS" placed at line start
WALLCLOCK_RE = re.compile(r"\b(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2})\b")
# - "[ 1234.56 ]" monotonic seconds in brackets (rare on some builds)
BRACKET_SEC_RE = re.compile(r"\[\s*([0-9]+(?:\.[0-9]+)?)\s*\]")


def parse_tegrastats_line(line: str) -> Dict[str, float]:
    """
    Extract rail -> instantaneous power (mW) from a line.
    Keeps only reasonable rail-like names (with '_' or common prefixes).
    """
    rails: Dict[str, float] = {}
    for name, val in RAIL_PAIR_RE.findall(line):
        if "_" in name or name.startswith(("VDD", "VIN", "POM", "TOTAL")):
            rails[name] = float(val)
    return rails


def parse_times_for_line(line: str) -> Optional[float]:
    """
    Parse an optional timestamp from a line.

    Returns:
        POSIX timestamp (float seconds) or None if not found/parsable.
    """
    m = WALLCLOCK_RE.search(line)
    if m:
        ts = m.group(1)
        try:
            dt = datetime.strptime(ts, "%m-%d-%Y %H:%M:%S")
            return dt.timestamp()
        except Exception:
            return None

    b = BRACKET_SEC_RE.search(line)
    if b:
        try:
            return float(b.group(1))
        except Exception:
            return None

    return None


def parse_log(filepath: str) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Parse a tegrastats log file into:
      - times_s: per-line seconds relative to the first timestamp (if available)
      - samples: list of dicts, each mapping rail->power_mW for that line

    If timestamps are inconsistent or missing, times_s is returned empty and callers
    should rely on dt_hint elsewhere.
    """
    times_abs: List[Optional[float]] = []
    samples: List[Dict[str, float]] = []

    # Read all raw lines (binary to robustly handle encoding)
    raw: List[str] = []
    with open(filepath, "rb") as f:
        for b in f:
            line = b.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            raw.append(line)

    if not raw:
        return [], []

    # Build absolute time list and per-line power samples
    for line in raw:
        times_abs.append(parse_times_for_line(line))
        rails = parse_tegrastats_line(line)
        if rails:
            samples.append(rails)

    if not samples:
        return [], []

    # Convert absolute times to relative seconds, if timestamps per raw line exist
    times_s: List[float] = []
    if len(times_abs) == len(raw) and all(t is not None for t in times_abs):
        t0 = times_abs[0]  # type: ignore
        times_s = [float(t - t0) for t in times_abs]  # type: ignore

        # If counts mismatch (e.g., some lines lacked rails), fall back to empty times
        if len(times_s) != len(samples):
            times_s = []

    return times_s, samples


# ---------- Energy integration ----------

def compute_energy(
    times_s: List[float],
    samples: List[Dict[str, float]],
    dt_hint_s: Optional[float] = None,
    force_fixed_dt: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Integrate per-rail energy from power samples.

    Parameters:
      times_s:
        Relative timestamps (s) per sample. Can be empty if not available.
      samples:
        List of rail->power_mW dicts (one per line).
      dt_hint_s:
        Sampling interval (s) to use when times_s is missing or when force_fixed_dt=True.
      force_fixed_dt:
        If True, ignore times_s and integrate using constant dt=dt_hint_s.

    Returns:
      rail -> {
        "avg_power_mW": average power over samples,
        "energy_J": integrated energy in joules,
        "energy_Wh": energy in watt-hours
      }
    """
    if not samples:
        return {}

    base_dt = dt_hint_s if dt_hint_s is not None else 0.2

    # Build per-sample dt list:
    # - If times_s is usable and not forced to fixed dt, use per-interval diffs.
    # - Else, use a constant dt (base_dt) for each sample.
    if (not force_fixed_dt) and times_s and len(times_s) >= len(samples):
        dts: List[float] = []
        for i in range(len(samples) - 1):
            dts.append(max(0.0, float(times_s[i + 1] - times_s[i])))
        dts.append(dts[len(dts)//2] if dts else base_dt)
    else:
        dts = [base_dt] * len(samples)

    # Collect all rails present across samples
    rails = sorted({k for s in samples for k in s.keys()})
    out: Dict[str, Dict[str, float]] = {}

    # Integrate mW * s to mJ, then derive J and Wh
    for r in rails:
        energy_mJ = 0.0
        sum_power = 0.0
        for i, s in enumerate(samples):
            p = float(s.get(r, 0.0))
            dt = dts[i] if i < len(dts) else dts[-1]
            energy_mJ += p * dt  # mW * s = mJ
            sum_power += p
        avg_power_mW = sum_power / max(1, len(samples))
        energy_J = energy_mJ / 1000.0
        energy_Wh = energy_J / 3600.0
        out[r] = {
            "avg_power_mW": avg_power_mW,
            "energy_J": energy_J,
            "energy_Wh": energy_Wh,
        }

    # Optional convenience: sum non-input rails (avoid double-counting input rails)
    if rails:
        sum_energy_J = 0.0
        sum_power_mW = 0.0
        for r in rails:
            if r in ("VIN_SYS_5V0", "POM_5V_IN", "TOTAL"):
                continue
            sum_energy_J += out[r]["energy_J"]
            sum_power_mW += out[r]["avg_power_mW"]
        if sum_energy_J > 0:
            out["SUM_NO_INPUT"] = {
                "avg_power_mW": sum_power_mW,
                "energy_J": sum_energy_J,
                "energy_Wh": sum_energy_J / 3600.0,
            }

    return out


def summarize_log(filepath: str, dt_hint_s: Optional[float] = None, force_fixed_dt: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Parse a log and return per-rail energy summary (see compute_energy()).

    force_fixed_dt=True is useful to guarantee each line contributes exactly dt_hint_s,
    avoiding any ambiguity from timestamps.
    """
    times_s, samples = parse_log(filepath)
    return compute_energy(times_s, samples, dt_hint_s=dt_hint_s, force_fixed_dt=force_fixed_dt)


def estimate_dt_from_interval_ms(interval_ms: int) -> float:
    """Convert tegrastats interval (ms) to seconds, clamped at 1 ms minimum."""
    return max(0.001, float(interval_ms) / 1000.0)


# ---------- CSV export ----------

def export_samples_csv(
    filepath_in: str,
    filepath_out: str,
    dt_hint_s: Optional[float] = None,
    rails_filter: Optional[List[str]] = None
) -> None:
    """
    Write time-series CSV: time_s + one column per rail (mW).

    If timestamps are missing, time is synthesized using dt_hint_s.
    rails_filter can restrict/ordering columns if provided.
    """
    times_s, samples = parse_log(filepath_in)
    if not samples:
        with open(filepath_out, "w", newline="") as f:
            csv.writer(f).writerow(["time_s"])
        return

    all_rails = sorted({k for s in samples for k in s.keys()})
    if rails_filter:
        rails = [r for r in rails_filter if r in all_rails]
    else:
        # Friendly ordering: input/total rails first
        priority = [r for r in ("POM_5V_IN", "VIN_SYS_5V0", "TOTAL") if r in all_rails]
        rest = sorted([r for r in all_rails if r not in priority])
        rails = priority + rest

    # Build a time axis if needed
    if not times_s or len(times_s) < len(samples):
        base_dt = dt_hint_s if dt_hint_s is not None else 0.2
        times_s = [i * base_dt for i in range(len(samples))]
    elif len(times_s) > len(samples):
        times_s = times_s[:len(samples)]

    with open(filepath_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s"] + rails)
        for i, s in enumerate(samples):
            row = [f"{times_s[i]:.6f}"] + [f"{float(s.get(r, 0.0)):.3f}" for r in rails]
            writer.writerow(row)


def export_summary_csv(
    filepath_in: str,
    filepath_out: str,
    dt_hint_s: Optional[float] = None,
    include_sum_no_input: bool = True
) -> None:
    """
    Write a per-rail summary CSV with columns:
      rail, avg_power_mW, energy_J, energy_Wh
    """
    summary = summarize_log(filepath_in, dt_hint_s=dt_hint_s)
    with open(filepath_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rail", "avg_power_mW", "energy_J", "energy_Wh"])
        for rail, vals in summary.items():
            if not include_sum_no_input and rail == "SUM_NO_INPUT":
                continue
            w.writerow([
                rail,
                f"{vals['avg_power_mW']:.3f}",
                f"{vals['energy_J']:.6f}",
                f"{vals['energy_Wh']:.9f}",
            ])


# ---------- Decorator that logs [Test, Energy] to its own CSV ----------

def measure_energy_to_csv(
    rail: str = "VIN_SYS_5V0",
    interval_ms: int = 200,
    log_dir: str = "powerlogs",
    num_runs: int = 3,
    guard_samples: int = 2,
    energy_csv_path: str = "energy_results.csv",
    append: bool = True,
    also_write_log_file: bool = True,
    fallback_rails: Optional[List[str]] = None
):
    """
    Decorator to measure energy around a function call and append [Test, Energy, Avg_Power_mW] to a CSV.

    Args:
        rail (str): The primary power rail to measure (e.g., "VIN_SYS_5V0").
        interval_ms (int): Sampling interval for tegrastats in milliseconds.
        log_dir (str): Base directory to store power logs.
        num_runs (int): Number of times to execute the function and average the results.
        guard_samples (int): Number of sampling intervals to wait before and after the function call.
        energy_csv_path (str): Path to the output CSV for energy results.
        append (bool): If True, append to the CSV; otherwise, create a new one.
        also_write_log_file (bool): If True, perform logging. If False, just run the function.
        fallback_rails (Optional[List[str]]): List of alternative rails to try if the primary is not found.

    Usage:
        @measure_energy_to_csv(...)
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
      - Appends a single row with the averaged results to `energy_csv_path`.
      - The 'Test' column in the CSV is annotated to show the number of runs averaged.
    """
    os.makedirs(log_dir, exist_ok=True)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract and remove the decorator-only tag argument to avoid upsetting the wrapped function
            tag = kwargs.pop("_bench_tag", "bench")
            dt_hint = estimate_dt_from_interval_ms(interval_ms)

            if not append and os.path.exists(energy_csv_path):
                os.remove(energy_csv_path)
            need_header = not os.path.exists(energy_csv_path)

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
            last_chosen_rail = rail
            any_samples_found = False
            result = None

            for i in range(num_runs):
                run_log_dir = os.path.join(log_dir, str(i + 1))
                os.makedirs(run_log_dir, exist_ok=True)
                log_path = os.path.join(run_log_dir, f"{tag}_{int(time.time())}.log")
                logger = TegrastatsLogger(log_path, interval_ms=interval_ms)

                try:
                    logger.start(append=False)
                    if guard_samples > 0:
                        time.sleep(dt_hint * guard_samples)
                    result = func(*args, **kwargs)
                    if guard_samples > 0:
                        time.sleep(dt_hint * guard_samples)
                finally:
                    logger.stop()

                line_count = 0
                if os.path.exists(log_path):
                    try:
                        with open(log_path, "r", encoding="utf-8", errors="ignore") as lf:
                            line_count = sum(1 for _ in lf)
                    except Exception:
                        pass
                
                if line_count == 0:
                    continue

                any_samples_found = True
                summary = summarize_log(log_path, dt_hint_s=dt_hint, force_fixed_dt=True)
                chosen_rail = rail
                vals = summary.get(chosen_rail)

                if not vals and fallback_rails:
                    for fr in fallback_rails:
                        if fr in summary:
                            chosen_rail = fr
                            vals = summary.get(fr)
                            break
                
                last_chosen_rail = chosen_rail
                if vals:
                    run_energies.append(float(vals.get("energy_J", 0.0)))
                    run_powers.append(float(vals.get("avg_power_mW", 0.0)))

            final_energy_J = 0.0
            final_avg_power_mW = 0.0
            tag_to_write = tag

            if run_energies:
                avg_energy = sum(run_energies) / len(run_energies)
                avg_power = sum(run_powers) / len(run_powers)
                final_energy_J = avg_energy
                final_avg_power_mW = avg_power
                tag_to_write = f"{tag}({last_chosen_rail}, avg of {len(run_energies)} runs)"
            elif any_samples_found:
                tag_to_write = f"{tag}(NO_RAIL)"
            else:
                tag_to_write = f"{tag}(NO_SAMPLES)"

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
    Simplest merge: append Energy and Avg_Power_mW from energy_results row i to benchmark row i.

    Assumes:
      - benchmark_csv_path has header:
          model,I,J,K,BATCH_SIZE,throughput_gops,latency_sec
      - energy_csv_path has header:
          Test,Energy,Avg_Power_mW
    Writes merged_csv_path with header:
      model,I,J,K,BATCH_SIZE,throughput_gops,latency_sec,Energy,Avg_Power_mW

    If the energy file has fewer rows than the benchmark, missing entries are filled with 0.
    """
    # Read all benchmark rows (including header)
    with open(benchmark_csv_path, "r", encoding="utf-8", errors="ignore") as fb:
        bench_reader = csv.reader(fb)
        bench_rows = list(bench_reader)

    if not bench_rows:
        raise ValueError("benchmark CSV is empty")

    # Read all energy rows (including header)
    with open(energy_csv_path, "r", encoding="utf-8", errors="ignore") as fe:
        energy_reader = csv.reader(fe)
        energy_rows = list(energy_reader)

    if not energy_rows:
        # No energy; copy benchmark and add zeros
        header = bench_rows[0] + ["Energy", "Avg_Power_mW"]
        with open(merged_csv_path, "w", newline="") as fo:
            writer = csv.writer(fo)
            writer.writerow(header)
            for r in bench_rows[1:]:
                writer.writerow(r + ["0.000000", "0.000"])
        return

    # Assume both have headers in first row
    bench_header = bench_rows[0]
    energy_header = energy_rows[0]

    # Find the "Energy" column index in the energy CSV header
    try:
        energy_idx = energy_header.index("Energy_J")
        power_idx = energy_header.index("Avg_Power_mW")
    except ValueError:
        raise ValueError("energy CSV must have columns named 'Energy' and 'Avg_Power_mW'")

    out_header = bench_header + ["Energy", "Avg_Power_mW"]
    out_rows = [out_header]

    # Align by row index (skip headers)
    bench_data = bench_rows[1:]
    energy_data = energy_rows[1:]

    for i, brow in enumerate(bench_data):
        if i < len(energy_data):
            erow = energy_data[i]
            # Defensive: if row is too short, default to 0
            energy_val = erow[energy_idx] if len(erow) > energy_idx else "0.000000"
            power_val = erow[power_idx] if len(erow) > power_idx else "0.000"
        else:
            energy_val = "0.000000"
            power_val = "0.000"
        out_rows.append(brow + [energy_val, power_val])

    # Write merged CSV
    with open(merged_csv_path, "w", newline="") as fo:
        writer = csv.writer(fo)
        writer.writerows(out_rows)


if __name__ == "__main__":
    # Simple CLI helper for ad-hoc tasks
    import argparse
    parser = argparse.ArgumentParser(description="tegrastats power logger + CSV + energy")
    sub = parser.add_subparsers(dest="cmd")

    # Start tegrastats
    p_start = sub.add_parser("start")
    p_start.add_argument("--log", required=True)
    p_start.add_argument("--interval-ms", type=int, default=200)
    p_start.add_argument("--overwrite", action="store_true")

    # Parse a log and print summary to stdout
    p_parse = sub.add_parser("parse")
    p_parse.add_argument("--log", required=True)
    p_parse.add_argument("--dt-hint", type=float, default=None)

    # Export time-series samples to CSV
    p_samples = sub.add_parser("samples-csv")
    p_samples.add_argument("--log", required=True)
    p_samples.add_argument("--out", required=True)
    p_samples.add_argument("--dt-hint", type=float, default=None)
    p_samples.add_argument("--rails", nargs="*", default=None)

    # Export summary CSV
    p_summary = sub.add_parser("summary-csv")
    p_summary.add_argument("--log", required=True)
    p_summary.add_argument("--out", required=True)
    p_summary.add_argument("--dt-hint", type=float, default=None)
    p_summary.add_argument("--no-sum", action="store_true")

    args = parser.parse_args()
    if args.cmd == "start":
        logger = TegrastatsLogger(args.log, interval_ms=args.interval_ms)
        logger.start(append=not args.overwrite)
        print(f"Started tegrastats -> {logger.log_path} at {args.interval_ms} ms")
    elif args.cmd == "parse":
        summary = summarize_log(args.log, dt_hint_s=args.dt_hint)
        print("Rail | Avg Power (mW) | Energy (J) | Energy (Wh)")
        print("-----------------------------------------------")
        for r, v in summary.items():
            print(f"{r} | {v['avg_power_mW']:.2f} | {v['energy_J']:.6f} | {v['energy_Wh']:.9f}")
    elif args.cmd == "samples-csv":
        export_samples_csv(args.log, args.out, dt_hint_s=args.dt_hint, rails_filter=args.rails)
        print(f"Wrote samples CSV -> {args.out}")
    elif args.cmd == "summary-csv":
        export_summary_csv(args.log, args.out, dt_hint_s=args.dt_hint, include_sum_no_input=not args.no_sum)
        print(f"Wrote summary CSV -> {args.out}")
    else:
        parser.print_help()

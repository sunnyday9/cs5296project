#!/usr/bin/env python3
"""
Record CPU, memory, and (if available) GPU usage on the machine where this script runs.
Intended to be run on the EC2 instance in a separate terminal/tmux pane while you
drive load from the client. Stop with Ctrl+C when the experiment finishes.

Usage on EC2 (from project root):
  python scripts/resource_logger.py --interval 1 --output results/resource_ec2_t3medium_xgb_n100.csv

Requires: psutil (pip install psutil). GPU stats use nvidia-smi (no extra Python package).
"""
import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import psutil
except ImportError:
    print("Install psutil: pip install psutil", file=sys.stderr)
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_gpu_stats():
    """Return (gpu_util_pct, gpu_mem_mb) or (None, None) if no NVIDIA GPU or nvidia-smi missing."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None, None
        line = out.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            util = int(parts[0].strip() or 0)
            mem_mb = int(parts[1].strip() or 0)
            return util, mem_mb
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Log CPU, memory, and optional GPU usage to CSV.")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: results/resource_<timestamp>.csv",
    )
    args = parser.parse_args()
    out_path = args.output
    if out_path is None:
        PROJECT_ROOT.joinpath("results").mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = PROJECT_ROOT / "results" / f"resource_{ts}.csv"
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    has_gpu = get_gpu_stats() != (None, None)
    fieldnames = ["timestamp_utc", "cpu_percent", "memory_percent"]
    if has_gpu:
        fieldnames.extend(["gpu_util_percent", "gpu_memory_mb"])

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        try:
            while True:
                ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory().percent
                row = {"timestamp_utc": ts, "cpu_percent": cpu, "memory_percent": mem}
                if has_gpu:
                    gpu_util, gpu_mem = get_gpu_stats()
                    row["gpu_util_percent"] = gpu_util if gpu_util is not None else ""
                    row["gpu_memory_mb"] = gpu_mem if gpu_mem is not None else ""
                w.writerow(row)
                f.flush()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass

    print(f"Resource log written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

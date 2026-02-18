#!/usr/bin/env python3
"""
Aggregate latency CSVs from results/ and compute cost estimates (EC2 + Lambda).
Output: summary CSV and optional figures. Use for report tables and tradeoff plots.
"""
import argparse
import csv
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Example pricing (replace with your region/instance; source: AWS pricing)
EC2_HOURLY_USD = {"t3.medium": 0.0416, "t3.large": 0.0832}
LAMBDA_PRICE_PER_1M = 0.20
LAMBDA_GB_SEC_PER_1M_GB_SEC = 0.0000166667  # $0.0000166667 per GB-second


def load_latency_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source"] = path.name
    return df


def aggregate_latencies(results_dir: Path) -> pd.DataFrame:
    frames = []
    for f in results_dir.glob("*_latencies.csv"):
        try:
            frames.append(load_latency_csv(f))
        except Exception as e:
            print(f"Skip {f}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def estimate_ec2_cost(instance_type: str, hours: float) -> float:
    return EC2_HOURLY_USD.get(instance_type, 0.05) * hours


def estimate_lambda_cost(num_invocations: int, avg_memory_gb: float, avg_duration_sec: float) -> float:
    request_cost = (num_invocations / 1_000_000) * LAMBDA_PRICE_PER_1M
    gb_sec = num_invocations * (avg_memory_gb * avg_duration_sec)
    gb_sec_cost = gb_sec * LAMBDA_GB_SEC_PER_1M_GB_SEC
    return request_cost + gb_sec_cost


def main():
    parser = argparse.ArgumentParser(description="Aggregate results and compute cost.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ec2-instance", default="t3.medium", help="Instance type for cost estimate")
    parser.add_argument("--ec2-hours", type=float, default=1.0, help="Assumed EC2 uptime (hours)")
    parser.add_argument("--lambda-memory-gb", type=float, default=1.0, help="Lambda memory in GB for cost")
    args = parser.parse_args()
    df = aggregate_latencies(args.results_dir)
    if df.empty:
        print("No latency CSVs found in", args.results_dir)
        return
    # Summary per source
    summary = df.groupby("source").agg(
        count=("latency_ms", "count"),
        mean_ms=("latency_ms", "mean"),
        p50=("latency_ms", lambda s: s.quantile(0.5)),
        p95=("latency_ms", lambda s: s.quantile(0.95)),
    ).round(2)
    if "cold" in df.columns:
        cold_first = df[df["cold"] == 1].groupby("source")["latency_ms"].mean()
        summary["cold_ms"] = cold_first
    # Cost placeholder (you can add actual EC2/Lambda tags to CSV or config)
    summary["cost_notes"] = ""
    for idx in summary.index:
        if "ec2" in idx.lower():
            summary.loc[idx, "cost_notes"] = f"EC2 {args.ec2_instance} ~${estimate_ec2_cost(args.ec2_instance, args.ec2_hours):.4f}/hr"
        elif "lambda" in idx.lower():
            n = summary.loc[idx, "count"]
            avg_ms = summary.loc[idx, "mean_ms"] or 100
            summary.loc[idx, "cost_notes"] = f"Lambda ~${estimate_lambda_cost(int(n), args.lambda_memory_gb, avg_ms / 1000):.6f} for {n} invocations"
    out_path = args.output or (args.results_dir / "summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path)
    print("Summary written to", out_path)
    print(summary)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Drive EC2 and/or Lambda experiments from a config file; write latencies to results/.
Example: python run_experiments.py --config experiment_config.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def run_ec2_trial(url: str, num_requests: int, tag: str) -> Path:
    out = RESULTS_DIR / f"ec2_{tag}_latencies.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable,
        PROJECT_ROOT / "scripts" / "client_ec2.py",
        "--url", url,
        "--num-requests", str(num_requests),
        "--output", str(out),
    ], check=True, cwd=PROJECT_ROOT)
    return out


def run_lambda_trial(function_name: str, num_requests: int, tag: str, region: str) -> Path:
    out = RESULTS_DIR / f"lambda_{tag}_latencies.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable,
        PROJECT_ROOT / "scripts" / "client_lambda.py",
        "--function-name", function_name,
        "--num-requests", str(num_requests),
        "--output", str(out),
        "--region", region,
    ], check=True, cwd=PROJECT_ROOT)
    return out


def main():
    parser = argparse.ArgumentParser(description="Run experiments from config.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "scripts" / "experiment_config.json")
    parser.add_argument("--only", choices=["ec2", "lambda"], default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    results = []
    if args.only != "lambda" and config.get("ec2", {}).get("url"):
        for trial in config["ec2"].get("trials", [{"num_requests": 100, "tag": "default"}]):
            out = run_ec2_trial(
                config["ec2"]["url"],
                trial.get("num_requests", 100),
                trial.get("tag", "default"),
            )
            results.append(str(out))
    if args.only != "ec2" and config.get("lambda", {}).get("function_name"):
        for trial in config["lambda"].get("trials", [{"num_requests": 100, "tag": "default"}]):
            out = run_lambda_trial(
                config["lambda"]["function_name"],
                trial.get("num_requests", 100),
                trial.get("tag", "default"),
                config["lambda"].get("region", "us-east-1"),
            )
            results.append(str(out))
    print("Results:", results)


if __name__ == "__main__":
    main()

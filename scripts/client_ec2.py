#!/usr/bin/env python3
"""
Client for EC2 inference server: send a fixed number of requests, record latency to CSV.
Usage: python client_ec2.py --url http://<EC2_IP>:5000 --num-requests 100 --output results/ec2_latencies.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import requests


def get_default_data_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "test_inputs.npy"


def load_inputs(data_path: Path, max_samples: int):
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run scripts/prepare_data.py --dataset adult (or cancer) first."
        )
    X = np.load(data_path)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X[:max_samples].tolist()


def main():
    parser = argparse.ArgumentParser(description="EC2 inference client – record latencies to CSV.")
    parser.add_argument("--url", required=True, help="Base URL of EC2 server (e.g. http://1.2.3.4:5000)")
    parser.add_argument("--num-requests", type=int, default=100, help="Number of requests to send")
    parser.add_argument("--data", type=Path, default=None, help="Path to test_inputs.npy (default: data/test_inputs.npy)")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path (default: stdout)")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    data_path = args.data or get_default_data_path()
    inputs = load_inputs(data_path, args.num_requests)

    rows = []
    for i, inp in enumerate(inputs):
        payload = {"input": inp}
        try:
            r = requests.post(f"{base}/predict", json=payload, timeout=30)
            data = r.json()
            latency_ms = data.get("latency_ms") or (r.elapsed.total_seconds() * 1000)
            rows.append({"request_id": i, "latency_ms": latency_ms, "status": r.status_code, "cold": 1 if i == 0 else 0})
        except Exception as e:
            rows.append({"request_id": i, "latency_ms": None, "status": "error", "cold": 1 if i == 0 else 0})
            print(f"Request {i} failed: {e}", file=sys.stderr)

    out_path = args.output
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["request_id", "latency_ms", "status", "cold"])
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out_path}")
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=["request_id", "latency_ms", "status", "cold"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

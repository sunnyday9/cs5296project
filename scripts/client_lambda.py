#!/usr/bin/env python3
"""
Lambda invoker: invoke function N times, record duration and billed duration to CSV.
Usage: python client_lambda.py --function-name ml-inference-group3 --num-requests 100 --output results/lambda_latencies.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import boto3
import numpy as np


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
    parser = argparse.ArgumentParser(description="Lambda invoker – record duration and cost to CSV.")
    parser.add_argument("--function-name", required=True, help="Lambda function name")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()
    data_path = args.data or get_default_data_path()
    inputs = load_inputs(data_path, args.num_requests)
    client = boto3.client("lambda", region_name=args.region)

    t_start = time.perf_counter()
    rows = []
    for i, inp in enumerate(inputs):
        payload = json.dumps({"input": inp})
        try:
            resp = client.invoke(
                FunctionName=args.function_name,
                InvocationType="RequestResponse",
                Payload=payload,
            )
            # Read response
            body = json.loads(resp["Payload"].read().decode())
            # Lambda returns body as string when via API; if invoked directly it's dict
            if isinstance(body.get("body"), str):
                inner = json.loads(body["body"])
            else:
                inner = body.get("body") or body
            latency_ms = inner.get("latency_ms")
            # Billed duration is in ms in response (optional)
            duration_ms = resp.get("ResponseMetadata", {}).get("HTTPHeaders", {}).get("x-amzn-requestid")
            # Actual duration from Lambda logs or use latency_ms as proxy
            billed_ms = resp.get("StatusCode") == 200 and (latency_ms or 0) or None
            rows.append({
                "request_id": i,
                "latency_ms": latency_ms,
                "status": resp["StatusCode"],
                "cold": 1 if i == 0 else 0,
            })
        except Exception as e:
            rows.append({"request_id": i, "latency_ms": None, "status": "error", "cold": 1 if i == 0 else 0})
            print(f"Request {i} failed: {e}", file=sys.stderr)
    total_wall_clock_sec = time.perf_counter() - t_start

    out_path = args.output
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["request_id", "latency_ms", "status", "cold"])
            w.writeheader()
            w.writerows(rows)
        summary = {
            "num_requests": len(rows),
            "total_wall_clock_sec": round(total_wall_clock_sec, 3),
            "latency_csv": out_path.name,
        }
        summary_path = out_path.parent / (out_path.stem + "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {len(rows)} rows to {out_path}")
        print(f"Total experiment time: {total_wall_clock_sec:.3f} s -> {summary_path}")
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=["request_id", "latency_ms", "status", "cold"])
        w.writeheader()
        w.writerows(rows)
        print(f"Total experiment time: {total_wall_clock_sec:.3f} s", file=sys.stderr)


if __name__ == "__main__":
    main()

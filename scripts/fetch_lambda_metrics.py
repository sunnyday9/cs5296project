#!/usr/bin/env python3
"""
Fetch Lambda CloudWatch metrics (Invocations, Duration) for a time range.
Run this after a Lambda experiment to record resource usage from AWS.

Usage:
  python scripts/fetch_lambda_metrics.py --function-name ml-inference-group3 --region us-east-1
  python scripts/fetch_lambda_metrics.py --function-name ml-inference-group3 --start "2025-02-21 10:00:00" --end "2025-02-21 10:05:00" --output results/lambda_metrics.json
"""
import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import boto3

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_naive_dt(s: str) -> datetime:
    """Parse 'YYYY-MM-DD HH:MM:SS' as UTC."""
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def main():
    parser = argparse.ArgumentParser(description="Fetch Lambda CloudWatch metrics for a time range.")
    parser.add_argument("--function-name", required=True, help="Lambda function name")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--start",
        default=None,
        help="Start time UTC, e.g. '2025-02-21 10:00:00'. Default: 5 minutes ago.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End time UTC. Default: now.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write JSON summary here")
    args = parser.parse_args()

    end = datetime.now(timezone.utc)
    if args.end:
        end = parse_naive_dt(args.end) if " " in args.end else datetime.fromisoformat(args.end.replace("Z", "+00:00"))
    start = end - timedelta(minutes=5)
    if args.start:
        start = parse_naive_dt(args.start) if " " in args.start else datetime.fromisoformat(args.start.replace("Z", "+00:00"))

    client = boto3.client("cloudwatch", region_name=args.region)
    name = args.function_name
    dim = [{"Name": "FunctionName", "Value": name}]
    period = 60

    inv = client.get_metric_statistics(
        Namespace="AWS/Lambda",
        MetricName="Invocations",
        Dimensions=dim,
        StartTime=start,
        EndTime=end,
        Period=period,
        Statistics=["Sum"],
    )
    duration = client.get_metric_statistics(
        Namespace="AWS/Lambda",
        MetricName="Duration",
        Dimensions=dim,
        StartTime=start,
        EndTime=end,
        Period=period,
        Statistics=["Average", "Sum", "Maximum", "Minimum", "SampleCount"],
    )

    total_invocations = sum(dp.get("Sum", 0) for dp in inv.get("Datapoints", []))
    datapoints_duration = duration.get("Datapoints", [])
    sum_duration_ms = sum(dp.get("Sum", 0) for dp in datapoints_duration)
    count_duration = sum(dp.get("SampleCount", 0) for dp in datapoints_duration)
    avg_duration_ms = (sum_duration_ms / count_duration) if count_duration else 0
    max_duration_ms = max((dp.get("Maximum", 0) for dp in datapoints_duration), default=0)
    min_duration_ms = min((dp.get("Minimum", 1e9) for dp in datapoints_duration), default=0)
    if min_duration_ms == 1e9:
        min_duration_ms = 0

    summary = {
        "function_name": name,
        "region": args.region,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "invocations_total": int(total_invocations),
        "duration_ms_sum": round(sum_duration_ms, 2),
        "duration_ms_avg": round(avg_duration_ms, 2),
        "duration_ms_min": round(min_duration_ms, 2),
        "duration_ms_max": round(max_duration_ms, 2),
        "sample_count": int(count_duration),
    }
    # Billed duration: Lambda rounds up each invocation to nearest 1 ms; total billed >= sum_duration_ms
    summary["billed_duration_ms_lower_bound"] = int(sum_duration_ms)

    print(json.dumps(summary, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.output}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()

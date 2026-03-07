from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import yaml


def load_prices(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "prices.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def latency_array(rows: list[dict], key: str = "latency_ms") -> np.ndarray:
    vals = [r[key] for r in rows if r.get(key) is not None and isinstance(r.get(key), (int, float))]
    return np.array(vals, dtype=np.float64) if vals else np.array([], dtype=np.float64)


def distribution_stats(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _cost_gb_sec_tiered(total_gb_sec: float, tiers: list[dict[str, Any]]) -> float:
    if not tiers or total_gb_sec <= 0:
        return 0.0
    cost = 0.0
    remaining = total_gb_sec
    prev_limit = 0.0
    for t in tiers:
        limit = t.get("limit_gb_sec")
        price = float(t.get("price_per_gb_sec", 0))
        if limit is None:
            cost += remaining * price
            break
        limit = float(limit)
        chunk = min(remaining, limit - prev_limit)
        if chunk > 0:
            cost += chunk * price
        remaining -= chunk
        prev_limit = limit
        if remaining <= 0:
            break
    return cost


def cost_lambda_from_latency_csv(
    rows: list[dict],
    price_per_million_requests: float,
    gb_second_tiers: list[dict[str, Any]] | None = None,
    price_per_gb_second: float | None = None,
    memory_mb_default: float = 128,
) -> dict[str, Any]:
    n = len(rows)
    total_gb_sec = 0.0
    for r in rows:
        lat = r.get("latency_ms")
        mem = r.get("memory_mb") or memory_mb_default
        if lat is not None and isinstance(lat, (int, float)):
            total_gb_sec += (mem / 1024.0) * (lat / 1000.0)
    cost_requests = (n / 1_000_000.0) * price_per_million_requests
    if gb_second_tiers:
        cost_duration = _cost_gb_sec_tiered(total_gb_sec, gb_second_tiers)
    else:
        cost_duration = total_gb_sec * (price_per_gb_second or 0.0)
    return {
        "num_requests": n,
        "total_gb_seconds": round(total_gb_sec, 6),
        "cost_requests_usd": round(cost_requests, 6),
        "cost_duration_usd": round(cost_duration, 6),
        "cost_total_usd": round(cost_requests + cost_duration, 6),
    }


def cost_lambda_from_metrics_json(
    metrics: dict[str, Any],
    price_per_million_requests: float,
    gb_second_tiers: list[dict[str, Any]] | None = None,
    price_per_gb_second: float | None = None,
    memory_mb: float = 128,
) -> dict[str, Any]:
    n = metrics.get("invocations_total") or metrics.get("sample_count") or 0
    total_ms = metrics.get("billed_duration_ms_lower_bound") or metrics.get("duration_ms_sum") or 0
    total_gb_sec = (memory_mb / 1024.0) * (total_ms / 1000.0)
    cost_requests = (n / 1_000_000.0) * price_per_million_requests
    if gb_second_tiers:
        cost_duration = _cost_gb_sec_tiered(total_gb_sec, gb_second_tiers)
    else:
        cost_duration = total_gb_sec * (price_per_gb_second or 0.0)
    return {
        "num_requests": n,
        "total_gb_seconds": round(total_gb_sec, 6),
        "cost_requests_usd": round(cost_requests, 6),
        "cost_duration_usd": round(cost_duration, 6),
        "cost_total_usd": round(cost_requests + cost_duration, 6),
    }


def cost_ec2_hours(
    experiment_duration_sec: float,
    price_per_hour: float,
    ebs_gb: float = 0,
    ebs_price_per_gb_month: float = 0,
) -> dict[str, Any]:
    hours = experiment_duration_sec / 3600.0
    cost_instance = hours * price_per_hour
    cost_ebs = 0.0
    if ebs_gb > 0 and ebs_price_per_gb_month > 0:
        hours_per_month = 24.0 * 30.0
        cost_ebs = (ebs_gb * ebs_price_per_gb_month) * (hours / hours_per_month)
    return {
        "duration_sec": experiment_duration_sec,
        "duration_hours": round(hours, 6),
        "price_per_hour_usd": price_per_hour,
        "cost_instance_usd": round(cost_instance, 6),
        "cost_ebs_usd": round(cost_ebs, 6),
        "cost_total_usd": round(cost_instance + cost_ebs, 6),
    }


def run_analysis(
    results_root: Path,
    prices_config_path: Path | None = None,
) -> dict[str, Any]:
    from .load_data import load_all_from_results

    data = load_all_from_results(results_root)
    prices = load_prices(prices_config_path)
    lambda_p = prices.get("lambda", {})
    ec2_p = prices.get("ec2", {})
    price_per_million = lambda_p.get("price_per_million_requests", 0.20)
    price_per_ms_at_128 = lambda_p.get("price_per_ms_at_128_mb")
    _price_per_gb_s_from_128 = (float(price_per_ms_at_128) * 8000.0) if price_per_ms_at_128 is not None else None
    gb_second_tiers = lambda_p.get("gb_second_tiers")
    if gb_second_tiers and _price_per_gb_s_from_128 is not None:
        gb_second_tiers = [{**t, "price_per_gb_sec": _price_per_gb_s_from_128} if i == 0 else t for i, t in enumerate(gb_second_tiers)]
    price_per_gb_s = lambda_p.get("price_per_gb_second") or _price_per_gb_s_from_128 or 0.0000166667
    if gb_second_tiers:
        price_per_gb_s = None
    ec2_by_type = ec2_p.get("price_per_hour_by_type") or {}
    ec2_default_hour = ec2_p.get("default_price_per_hour", 0.0416)
    ebs_gb = float(ec2_p.get("ebs_gp2_gb", 0) or 0)
    ebs_price_per_gb_month = float(ec2_p.get("ebs_gp2_price_per_gb_month", 0) or 0)

    report = {
        "lambda_latencies": [],
        "lambda_metrics": [],
        "ec2_latencies": [],
        "ec2_resources": [],
    }

    for item in data["lambda_latencies"]:
        rows = item["data"]
        lat_arr = latency_array(rows)
        cost = cost_lambda_from_latency_csv(
            rows,
            price_per_million,
            gb_second_tiers=gb_second_tiers,
            price_per_gb_second=price_per_gb_s,
            memory_mb_default=128,
        )
        summary_path = Path(item["path"]).parent / (Path(item["path"]).stem + "_summary.json")
        duration_sec = 0.0
        if summary_path.exists():
            try:
                with open(summary_path, encoding="utf-8") as f:
                    s = __import__("json").load(f)
                duration_sec = float(s.get("total_wall_clock_sec", 0))
            except Exception:
                pass
        if duration_sec <= 0 and lat_arr.size > 0:
            duration_sec = float(np.sum(lat_arr) / 1000.0)
        report["lambda_latencies"].append({
            "name": item["name"],
            "path": str(item["path"]),
            "duration_sec": duration_sec,
            "num_requests": cost["num_requests"],
            "distribution": distribution_stats(lat_arr),
            "cost": cost,
        })

    for item in data["lambda_metrics"]:
        m = item["data"]
        mem_mb = 128
        cost = cost_lambda_from_metrics_json(
            m, price_per_million,
            gb_second_tiers=gb_second_tiers,
            price_per_gb_second=price_per_gb_s,
            memory_mb=mem_mb,
        )
        report["lambda_metrics"].append({
            "name": item["name"],
            "path": str(item["path"]),
            "metrics": m,
            "cost": cost,
        })

    for item in data["ec2_latencies"]:
        rows = item["data"]
        lat_arr = latency_array(rows)
        instance_type = item.get("instance_type")
        price_hour = ec2_by_type.get(instance_type or "", ec2_default_hour)
        summary_path = item["path"].parent / (item["path"].stem + "_summary.json")
        duration_sec = 0.0
        if summary_path.exists():
            try:
                with open(summary_path, encoding="utf-8") as f:
                    s = __import__("json").load(f)
                duration_sec = float(s.get("total_wall_clock_sec", 0))
            except Exception:
                pass
        if duration_sec <= 0 and lat_arr.size > 0:
            duration_sec = float(np.sum(lat_arr) / 1000.0)
        cost = cost_ec2_hours(duration_sec, price_hour, ebs_gb=ebs_gb, ebs_price_per_gb_month=ebs_price_per_gb_month)
        report["ec2_latencies"].append({
            "name": item["name"],
            "path": str(item["path"]),
            "instance_type": instance_type,
            "distribution": distribution_stats(lat_arr),
            "duration_sec": duration_sec,
            "cost": cost,
        })

    for item in data["ec2_resources"]:
        rows = item["data"]
        instance_type = item.get("instance_type")
        price_hour = ec2_by_type.get(instance_type or "", ec2_default_hour)
        if not rows:
            duration_sec = 0.0
        else:
            try:
                from datetime import datetime
                t0 = datetime.fromisoformat(rows[0]["timestamp_utc"].replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(rows[-1]["timestamp_utc"].replace("Z", "+00:00"))
                duration_sec = (t1 - t0).total_seconds()
            except Exception:
                duration_sec = len(rows)
        cost = cost_ec2_hours(duration_sec, price_hour, ebs_gb=ebs_gb, ebs_price_per_gb_month=ebs_price_per_gb_month)
        cpu = np.array([r["cpu_percent"] for r in rows if r.get("cpu_percent") is not None], dtype=np.float64)
        mem = np.array([r["memory_percent"] for r in rows if r.get("memory_percent") is not None], dtype=np.float64)
        report["ec2_resources"].append({
            "name": item["name"],
            "path": str(item["path"]),
            "instance_type": instance_type,
            "duration_sec": duration_sec,
            "cost": cost,
            "cpu_distribution": distribution_stats(cpu) if cpu.size else {},
            "memory_distribution": distribution_stats(mem) if mem.size else {},
            "cpu_series": cpu.tolist(),
            "memory_series": mem.tolist(),
        })
    report["_loaded_data"] = {
        "lambda_latencies": [{"name": item["name"], "data": item["data"]} for item in data["lambda_latencies"]],
    }
    report["_loaded_data"]["ec2_latencies"] = [
        {"name": item["name"], "data": item["data"]} for item in data["ec2_latencies"]
    ]
    return report

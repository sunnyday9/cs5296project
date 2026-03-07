from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any
import numpy as np


def load_lambda_latency_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            for k in ("latency_ms", "memory_mb", "cold"):
                if k in row and row[k] not in ("", None):
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        row[k] = None
            if "request_id" in row and row["request_id"] not in ("", None):
                try:
                    row["request_id"] = int(row["request_id"])
                except ValueError:
                    pass
            rows.append(row)
    return rows


def load_lambda_metrics_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_ec2_latency_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            for k in ("latency_ms", "cold"):
                if k in row and row[k] not in ("", None):
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        row[k] = None
            if "request_id" in row and row["request_id"] not in ("", None):
                try:
                    row["request_id"] = int(row["request_id"])
                except ValueError:
                    pass
            rows.append(row)
    return rows


def load_resource_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            for k in ("cpu_percent", "memory_percent", "gpu_util_percent", "gpu_memory_mb"):
                if k in row and row[k] not in ("", None):
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        row[k] = None
            rows.append(row)
    return rows


def instance_type_from_folder_name(folder_name: str) -> str:
    name = folder_name.strip().lower().replace("-", "")
    if name == "t2micro":
        return "t2.micro"
    if name == "t3micro":
        return "t3.micro"
    if name == "t2medium":
        return "t2.medium"
    if name.startswith("t3") and len(name) > 2:
        return f"t3.{name[2:]}"
    if name.startswith("t2") and len(name) > 2:
        return f"t2.{name[2:]}"
    return name


def infer_ec2_instance_type_from_path(path: Path) -> str | None:
    name = path.stem.lower()
    for part in name.replace("-", "").split("_"):
        if part.startswith("t3") and part != "t3":
            return "t3.micro" if part == "t3micro" else ("t3.medium" if part == "t3medium" else f"t3.{part[2:]}" if len(part) > 2 else "t3.micro")
        if part.startswith("t2"):
            return "t2.micro" if part == "t2micro" else ("t2.medium" if part == "t2medium" else f"t2.{part[2:]}" if len(part) > 2 else "t2.micro")
    return None


def load_all_from_results(results_root: Path) -> dict[str, Any]:
    results_root = Path(results_root)
    out = {
        "lambda_latencies": [],
        "lambda_metrics": [],
        "ec2_latencies": [],
        "ec2_resources": [],
    }

    lambda_dir = results_root / "lambda"
    if lambda_dir.is_dir():
        for p in lambda_dir.iterdir():
            if p.suffix.lower() == ".csv" and "latenc" in p.name.lower():
                out["lambda_latencies"].append({
                    "path": p,
                    "name": p.stem,
                    "data": load_lambda_latency_csv(p),
                })
            if p.suffix.lower() == ".json" and "metric" in p.name.lower():
                out["lambda_metrics"].append({
                    "path": p,
                    "name": p.stem,
                    "data": load_lambda_metrics_json(p),
                })
    latencies_dir = results_root / "latencies"
    if latencies_dir.is_dir():
        subdirs = [d for d in latencies_dir.iterdir() if d.is_dir()]
        if subdirs:
            for subdir in subdirs:
                instance_type = instance_type_from_folder_name(subdir.name)
                for p in subdir.iterdir():
                    if p.is_file() and p.suffix.lower() == ".csv" and "summary" not in p.name.lower():
                        out["ec2_latencies"].append({
                            "path": p,
                            "name": p.stem,
                            "instance_type": instance_type,
                            "data": load_ec2_latency_csv(p),
                        })
        else:
            for p in latencies_dir.iterdir():
                if p.is_file() and p.suffix.lower() == ".csv":
                    out["ec2_latencies"].append({
                        "path": p,
                        "name": p.stem,
                        "instance_type": infer_ec2_instance_type_from_path(p),
                        "data": load_ec2_latency_csv(p),
                    })
    resources_dir = results_root / "resources"
    if resources_dir.is_dir():
        subdirs = [d for d in resources_dir.iterdir() if d.is_dir()]
        if subdirs:
            for subdir in subdirs:
                instance_type = instance_type_from_folder_name(subdir.name)
                for p in subdir.iterdir():
                    if p.is_file() and p.suffix.lower() == ".csv" and "resource" in p.name.lower():
                        out["ec2_resources"].append({
                            "path": p,
                            "name": p.stem,
                            "instance_type": instance_type,
                            "data": load_resource_csv(p),
                        })
        else:
            for p in resources_dir.iterdir():
                if p.is_file() and p.suffix.lower() == ".csv" and "resource" in p.name.lower():
                    out["ec2_resources"].append({
                        "path": p,
                        "name": p.stem,
                        "instance_type": infer_ec2_instance_type_from_path(p),
                        "data": load_resource_csv(p),
                    })
    if not out["lambda_latencies"] and not out["lambda_metrics"]:
        for p in results_root.iterdir():
            if p.is_file() and p.suffix.lower() == ".csv" and "lambda" in p.name.lower() and "latenc" in p.name.lower():
                out["lambda_latencies"].append({"path": p, "name": p.stem, "data": load_lambda_latency_csv(p)})
            if p.is_file() and p.suffix.lower() == ".json" and "lambda" in p.name.lower() and "metric" in p.name.lower():
                out["lambda_metrics"].append({"path": p, "name": p.stem, "data": load_lambda_metrics_json(p)})
    if not out["ec2_latencies"]:
        for p in results_root.iterdir():
            if p.is_file() and p.suffix.lower() == ".csv" and ("ec2" in p.name.lower() or "latenc" in p.name.lower()) and "resource" not in p.name.lower():
                out["ec2_latencies"].append({
                    "path": p,
                    "name": p.stem,
                    "instance_type": infer_ec2_instance_type_from_path(p),
                    "data": load_ec2_latency_csv(p),
                })
    if not out["ec2_resources"]:
        for p in results_root.iterdir():
            if p.is_file() and p.suffix.lower() == ".csv" and "resource" in p.name.lower():
                out["ec2_resources"].append({
                    "path": p,
                    "name": p.stem,
                    "instance_type": infer_ec2_instance_type_from_path(p),
                    "data": load_resource_csv(p),
                })
    return out

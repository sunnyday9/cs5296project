from __future__ import annotations
import re
from pathlib import Path
from typing import Any
import numpy as np


def _name_to_experiment_label(name: str) -> str:
    name_l = name.lower()
    dataset = "unknown"
    if "_adult_" in name_l or name_l.startswith("adult_") or name_l.endswith("_adult"):
        dataset = "adult"
    elif "_cancer_" in name_l or name_l.startswith("cancer_") or name_l.endswith("_cancer"):
        dataset = "cancer"
    else:
        for token in ("adult", "cancer"):
            if token in name_l:
                dataset = token
                break
    model = "unknown"
    if "_mlp_" in name_l or "_mlp" in name_l or "mlp_" in name_l:
        model = "mlp"
    elif "_xgb_" in name_l or "_xgb" in name_l or "xgb_" in name_l:
        model = "xgb"
    else:
        for token in ("mlp", "xgb"):
            if token in name_l:
                model = token
                break
    match = re.search(r"_n(\d+)(_|$)", name_l)
    n = match.group(1) if match else "?"
    return f"{dataset}-{model}-{n}"

def _ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def plot_latency_distribution(
    latency_arrays: list[tuple[str, np.ndarray]],
    title: str = "Latency distribution",
    output_path: Path | None = None,
) -> None:
    plt = _ensure_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    all_vals = []
    for label, arr in latency_arrays:
        arr = np.asarray(arr)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        all_vals.append(arr)
        ax1.hist(arr, bins=min(50, max(10, arr.size // 5)), alpha=0.5, label=label, density=True)
        s = np.sort(arr)
        cdf = np.arange(1, len(s) + 1, dtype=float) / len(s)
        ax2.plot(s, cdf, label=label)
    if all_vals:
        combined = np.concatenate(all_vals)
        x_max = float(np.percentile(combined, 98))
        x_max = max(2.0, min(x_max * 1.1, 200.0))
        ax1.set_xlim(left=0, right=x_max)
        ax2.set_xlim(left=0, right=x_max)
    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Density")
    ax1.set_title("Histogram")
    ax1.legend(fontsize=7)
    ax2.set_xlabel("Latency (ms)")
    ax2.set_ylabel("CDF")
    ax2.set_title("Cumulative distribution")
    ax2.legend(fontsize=7)
    fig.suptitle(title)
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_resource_time_series(
    timestamps: list[str],
    cpu_percent: list[float],
    memory_percent: list[float],
    title: str = "EC2 resource usage",
    output_path: Path | None = None,
) -> None:
    plt = _ensure_matplotlib()
    x = np.arange(len(cpu_percent)) if not timestamps else np.arange(len(timestamps))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(x, cpu_percent, color="C0", alpha=0.8)
    ax1.set_ylabel("CPU %")
    ax1.set_title("CPU utilization")
    ax1.grid(True, alpha=0.3)
    ax2.plot(x, memory_percent, color="C1", alpha=0.8)
    ax2.set_ylabel("Memory %")
    ax2.set_xlabel("Sample index")
    ax2.set_title("Memory utilization")
    ax2.grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _resource_group_key_and_n(item: dict[str, Any]) -> tuple[tuple[str, str, str], int, str]:
    name = item.get("name", "")
    instance_type = item.get("instance_type") or "unknown"
    label = _name_to_experiment_label(name)
    parts = label.split("-")
    dataset = parts[0] if len(parts) > 0 else "unknown"
    model = parts[1] if len(parts) > 1 else "unknown"
    n_str = parts[2] if len(parts) > 2 else "?"
    try:
        n_int = int(n_str)
    except ValueError:
        n_int = 0
    n_label = f"n{n_str}"
    return (instance_type, dataset, model), n_int, n_label


def _pad_series(series: list, target_len: int, default: float = 0.0) -> list:
    if len(series) >= target_len:
        return list(series[:target_len])
    if not series:
        return [default] * target_len
    return list(series) + [series[-1]] * (target_len - len(series))


def plot_resource_time_series_grouped(
    report: dict[str, Any],
    output_dir: Path,
) -> None:
    items = report.get("ec2_resources", [])
    if not items:
        return
    by_instance: dict[str, dict[tuple[str, str], list[tuple[int, str, dict]]]] = {}
    for item in items:
        (instance_type, dataset, model), n_int, n_label = _resource_group_key_and_n(item)
        if instance_type not in by_instance:
            by_instance[instance_type] = {}
        key_dm = (dataset, model)
        if key_dm not in by_instance[instance_type]:
            by_instance[instance_type][key_dm] = []
        by_instance[instance_type][key_dm].append((n_int, n_label, item))
    col_order = [("adult", "mlp"), ("adult", "xgb"), ("cancer", "mlp"), ("cancer", "xgb")]
    output_dir = Path(output_dir)
    plt = _ensure_matplotlib()
    for instance_type, dm_groups in by_instance.items():
        fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=False)
        if axes.ndim == 1:
            axes = axes.reshape(2, -1)
        for col_idx, (dataset, model) in enumerate(col_order):
            group_items = dm_groups.get((dataset, model), [])
            group_items = sorted(group_items, key=lambda x: x[0])
            ax_cpu, ax_mem = axes[0, col_idx], axes[1, col_idx]
            for n_int, n_label, item in group_items:
                cpu = item.get("cpu_series") or []
                mem = item.get("memory_series") or []
                if not cpu and not mem:
                    continue
                if not cpu:
                    cpu = [0.0] * len(mem) if mem else [0.0]
                if not mem:
                    mem = [0.0] * len(cpu) if cpu else [0.0]
                n_len = max(len(cpu), len(mem))
                cpu = _pad_series(cpu, n_len)
                mem = _pad_series(mem, n_len)
                x = np.arange(n_len)
                ax_cpu.plot(x, cpu, alpha=0.8, label=n_label)
                ax_mem.plot(x, mem, alpha=0.8, label=n_label)
            ax_cpu.set_ylabel("CPU %")
            ax_cpu.set_title(f"{dataset}-{model}")
            ax_cpu.legend(fontsize=7)
            ax_cpu.grid(True, alpha=0.3)
            ax_mem.set_ylabel("Memory %")
            ax_mem.set_xlabel("Sample index")
            ax_mem.legend(fontsize=7)
            ax_mem.grid(True, alpha=0.3)
        y_cpu_lo, y_cpu_hi = float("inf"), float("-inf")
        y_mem_lo, y_mem_hi = float("inf"), float("-inf")
        for ax in axes[0, :]:
            lo, hi = ax.get_ylim()
            y_cpu_lo = min(y_cpu_lo, lo)
            y_cpu_hi = max(y_cpu_hi, hi)
        for ax in axes[1, :]:
            lo, hi = ax.get_ylim()
            y_mem_lo = min(y_mem_lo, lo)
            y_mem_hi = max(y_mem_hi, hi)
        margin = 0.05
        if y_cpu_hi > y_cpu_lo:
            pad = (y_cpu_hi - y_cpu_lo) * margin
            y_cpu_lo = max(0, y_cpu_lo - pad)
            y_cpu_hi = min(100, y_cpu_hi + pad)
        else:
            y_cpu_lo, y_cpu_hi = 0, 100
        if y_mem_hi > y_mem_lo:
            pad = (y_mem_hi - y_mem_lo) * margin
            y_mem_lo = max(0, y_mem_lo - pad)
            y_mem_hi = min(100, y_mem_hi + pad)
        else:
            y_mem_lo, y_mem_hi = 0, 100
        for ax in axes[0, :]:
            ax.set_ylim(y_cpu_lo, y_cpu_hi)
        for ax in axes[1, :]:
            ax.set_ylim(y_mem_lo, y_mem_hi)
        fig.suptitle(f"Resource usage: {instance_type}")
        plt.tight_layout()
        out_path = output_dir / f"resource_{instance_type}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_cost_comparison(
    labels: list[str],
    costs_usd: list[float],
    title: str = "Cost comparison",
    output_path: Path | None = None,
) -> None:
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, costs_usd, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Cost (USD)")
    ax.set_title(title)
    for b, v in zip(bars, costs_usd):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0005, f"${v:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _sort_experiment_labels(labels: list[str]) -> list[str]:
    def key(s):
        parts = s.split("-")
        if len(parts) != 3:
            return (0, 0, 0)
        ds, model, n = parts
        n_val = int(n) if n.isdigit() else 0
        return (0 if ds == "adult" else 1, 0 if model == "xgb" else 1, n_val)
    return sorted(labels, key=key)


def plot_cost_comparison_by_instance(
    report: dict[str, Any],
    output_path: Path | None = None,
) -> None:
    plt = _ensure_matplotlib()
    groups = [
        ("Lambda", []),
        ("t2.medium", []),
        ("t3.micro", []),
        ("t2.micro", []),
    ]
    def _skip_label(label: str) -> bool:
        return "-?" in label or label.endswith("-?")

    group_map = {g[0]: g[1] for g in groups}
    for item in report.get("lambda_latencies", []) + report.get("lambda_metrics", []):
        label = _name_to_experiment_label(item.get("name", ""))
        if _skip_label(label):
            continue
        cost = item.get("cost", {}).get("cost_total_usd", 0)
        group_map["Lambda"].append((label, cost))
    for item in report.get("ec2_latencies", []):
        it = item.get("instance_type") or "unknown"
        if it not in group_map:
            group_map[it] = []
        label = _name_to_experiment_label(item.get("name", ""))
        if _skip_label(label):
            continue
        cost = item.get("cost", {}).get("cost_total_usd", 0)
        group_map[it].append((label, cost))
    for item in report.get("ec2_resources", []):
        it = item.get("instance_type") or "unknown"
        if it not in group_map:
            group_map[it] = []
        label = _name_to_experiment_label(item.get("name", ""))
        if _skip_label(label):
            continue
        cost = item.get("cost", {}).get("cost_total_usd", 0)
        group_map[it].append((label, cost))
    plot_groups = [(name, items) for name, items in groups if items]
    if not plot_groups:
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        return
    n_sub = len(plot_groups)
    n_col = min(2, n_sub)
    n_row = (n_sub + n_col - 1) // n_col
    fig, axes = plt.subplots(n_row, n_col, figsize=(6 * n_col, 4 * n_row))
    if n_sub == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for idx, (instance_name, items) in enumerate(plot_groups):
        ax = axes[idx]
        seen = {}
        for label, cost in items:
            if label not in seen:
                seen[label] = cost
        labels_sorted = _sort_experiment_labels(list(seen.keys()))
        costs = [seen[l] for l in labels_sorted]
        x = np.arange(len(labels_sorted))
        bars = ax.bar(x, costs, color="steelblue", edgecolor="navy", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_sorted, rotation=45, ha="right")
        ax.set_ylabel("Cost (USD)")
        ax.set_title(instance_name)
    for j in range(len(plot_groups), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Cost comparison by instance type")
    plt.tight_layout(rect=[0, 0.2, 1, 0.96])
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


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


def plot_cost_vs_request_density(
    report: dict[str, Any],
    output_path: Path | None = None,
    prices_config_path: Path | None = None,
) -> None:
    from .analyze import load_prices
    
    request_densities = [10, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    request_densities = np.array(request_densities, dtype=np.float64)

    prices = load_prices(prices_config_path)
    lambda_p = prices.get("lambda", {})
    ec2_p = prices.get("ec2", {})
    price_per_million = lambda_p.get("price_per_million_requests", 0.20)
    gb_second_tiers = lambda_p.get("gb_second_tiers")
    price_per_gb_s = lambda_p.get("price_per_gb_second", 0.0000166667) if not gb_second_tiers else None
    ec2_by_type = ec2_p.get("price_per_hour_by_type") or {}
    ebs_gb = float(ec2_p.get("ebs_gp2_gb", 0) or 0)
    ebs_price_per_gb_month = float(ec2_p.get("ebs_gp2_price_per_gb_month", 0) or 0)
    ebs_per_hour = (ebs_gb * ebs_price_per_gb_month) / (24.0 * 30.0) if (ebs_gb and ebs_price_per_gb_month) else 0.0
    avg_gb_sec_per_request = 0.0
    for item in report.get("lambda_latencies", []):
        c = item.get("cost", {})
        n, gb = c.get("num_requests", 0), c.get("total_gb_seconds", 0)
        if n and gb > 0:
            avg_gb_sec_per_request = gb / n
            break
    if avg_gb_sec_per_request <= 0:
        avg_gb_sec_per_request = (128.0 / 1024.0) * (100.0 / 1000.0)
    lambda_costs = []
    for r in request_densities:
        cost_req = (r / 1e6) * price_per_million
        total_gb_sec = r * avg_gb_sec_per_request
        if gb_second_tiers:
            cost_dur = _cost_gb_sec_tiered(total_gb_sec, gb_second_tiers)
        else:
            cost_dur = total_gb_sec * (price_per_gb_s or 0)
        lambda_costs.append(cost_req + cost_dur)

    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"Lambda": "C0", "EC2 t2.medium": "C1", "EC2 t3.micro": "C2", "EC2 t2.micro": "C3"}
    markers = {"Lambda": "o", "EC2 t2.medium": "s", "EC2 t3.micro": "^", "EC2 t2.micro": "D"}

    ax.plot(
        request_densities,
        lambda_costs,
        label="Lambda",
        color=colors["Lambda"],
        marker=markers["Lambda"],
        markersize=6,
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
    )

    for instance_key, label in [("t2.medium", "EC2 t2.medium"), ("t3.micro", "EC2 t3.micro"), ("t2.micro", "EC2 t2.micro")]:
        price_hour = ec2_by_type.get(instance_key)
        if price_hour is None:
            continue
        cost_per_hour = price_hour + ebs_per_hour
        ec2_costs = np.full_like(request_densities, cost_per_hour, dtype=np.float64)
        ax.plot(
            request_densities,
            ec2_costs,
            label=label,
            color=colors.get(label, None),
            marker=markers.get(label, "o"),
            markersize=6,
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
        )

    ax.set_xlabel("Request density (requests / hour)")
    ax.set_ylabel("Cost (USD)")
    ax.set_title("Cost vs request density: Lambda vs EC2 instances")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _latency_arrays_from_rows(rows: list[dict], key: str = "latency_ms") -> np.ndarray:
    vals = [r[key] for r in rows if r.get(key) is not None and isinstance(r.get(key), (int, float))]
    return np.array(vals, dtype=np.float64) if vals else np.array([], dtype=np.float64)


def plot_ec2_latency_distribution_by_instance(
    report: dict[str, Any],
    output_path: Path | None = None,
) -> None:
    loaded = report.get("_loaded_data") or {}
    ec2_loaded = loaded.get("ec2_latencies", [])
    ec2_report = {item["name"]: item for item in report.get("ec2_latencies", [])}
    if not ec2_loaded:
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        return
    by_instance = {}
    for item in ec2_loaded:
        name = item.get("name", "")
        rec = ec2_report.get(name, {})
        it = rec.get("instance_type") or "unknown"
        if it not in by_instance:
            by_instance[it] = []
        arr = _latency_arrays_from_rows(item.get("data", []))
        if arr.size > 0:
            by_instance[it].append((_name_to_experiment_label(name), arr))
    if not by_instance:
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        return
    order = ["t2.medium", "t3.micro", "t2.micro"]
    plot_groups = [(k, by_instance[k]) for k in order if k in by_instance]
    for k in by_instance:
        if k not in order:
            plot_groups.append((k, by_instance[k]))
    n_sub = len(plot_groups)
    plt = _ensure_matplotlib()
    fig, axes2 = plt.subplots(n_sub, 2, figsize=(12, 4 * n_sub))
    if n_sub == 1:
        axes2 = axes2.reshape(1, -1)
    for idx, (instance_name, series) in enumerate(plot_groups):
        ax1, ax2 = axes2[idx, 0], axes2[idx, 1]
        all_vals = []
        for label, arr in series:
            arr = np.asarray(arr)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            all_vals.append(arr)
            ax1.hist(arr, bins=min(50, max(10, arr.size // 5)), alpha=0.5, label=label, density=True)
            s = np.sort(arr)
            cdf = np.arange(1, len(s) + 1, dtype=float) / len(s)
            ax2.plot(s, cdf, label=label)
        if all_vals:
            combined = np.concatenate(all_vals)
            x_max = float(np.percentile(combined, 98))
            x_max = max(2.0, min(x_max * 1.1, 100.0))
            ax1.set_xlim(left=0, right=x_max)
            ax2.set_xlim(left=0, right=x_max)
        ax1.set_xlabel("Latency (ms)")
        ax1.set_ylabel("Density")
        ax1.set_title(f"{instance_name} – Histogram")
        ax1.legend(fontsize=7)
        ax2.set_xlabel("Latency (ms)")
        ax2.set_ylabel("CDF")
        ax2.set_title(f"{instance_name} – CDF")
        ax2.legend(fontsize=7)
    fig.suptitle("EC2 latency distribution by instance type")
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def generate_all_plots(
    report: dict[str, Any],
    output_dir: Path,
    latency_data: dict[str, Any] | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = report.get("_loaded_data") or latency_data or {}
    if loaded:
        lambda_items = loaded.get("lambda_latencies", [])
        if lambda_items:
            arrs = [(item["name"], _latency_arrays_from_rows(item["data"])) for item in lambda_items]
            arrs = [(n, a) for n, a in arrs if a.size > 0]
            if arrs:
                plot_latency_distribution(
                    arrs,
                    title="Lambda latency distribution",
                    output_path=output_dir / "lambda_latency_distribution.png",
                )
        ec2_items = loaded.get("ec2_latencies", [])
        if ec2_items:
            plot_ec2_latency_distribution_by_instance(report, output_path=output_dir / "ec2_latency_distribution.png")

    plot_cost_comparison_by_instance(report, output_path=output_dir / "cost_comparison.png")
    plot_cost_vs_request_density(report, output_path=output_dir / "cost_vs_request_density.png")
    plot_resource_time_series_grouped(report, output_dir)

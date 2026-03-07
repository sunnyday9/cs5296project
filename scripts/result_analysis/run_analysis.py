from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    _project_root = Path(__file__).resolve().parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from scripts.result_analysis.analyze import run_analysis
from scripts.result_analysis.plots import generate_all_plots

def _strip_loaded_data(report: dict) -> dict:
    """Return a copy of report without _loaded_data for JSON serialization."""
    out = {k: v for k, v in report.items() if k != "_loaded_data"}
    return out

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Lambda/EC2 experiment results: distributions, cost, and plots.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "results",
        help="Root directory containing lambda/, latencies/, resources/ (default: project results/)",
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=None,
        help="Path to prices YAML (default: scripts/result_analysis/prices.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary JSON and plots (default: results_dir/analysis)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plot images",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip writing summary JSON",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or (results_dir / "analysis")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prices_path = args.prices
    if prices_path is not None:
        prices_path = prices_path.resolve()
        if not prices_path.is_file():
            print(f"Prices file not found: {prices_path}", file=sys.stderr)
            sys.exit(1)

    print("Loading and analyzing results from", results_dir)
    report = run_analysis(results_dir, prices_config_path=prices_path)

    if not args.no_json:
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_strip_loaded_data(report), f, indent=2)
        print("Wrote", summary_path)

    if not args.no_plots:
        generate_all_plots(report, output_dir)
        print("Plots written to", output_dir)

    print("\n--- Cost summary ---")
    for item in report.get("lambda_latencies", []) + report.get("lambda_metrics", []):
        c = item.get("cost", {})
        print(f"  {item.get('name', '')}: ${c.get('cost_total_usd', 0):.6f} ({c.get('num_requests', 0)} requests)")
    for item in report.get("ec2_latencies", []) + report.get("ec2_resources", []):
        c = item.get("cost", {})
        print(f"  {item.get('name', '')}: ${c.get('cost_total_usd', 0):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

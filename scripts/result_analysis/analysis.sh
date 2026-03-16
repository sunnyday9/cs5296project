# Default: results dir = project/results, output = results/analysis
python scripts/result_analysis/run_analysis.py

# Custom paths
python scripts/result_analysis/run_analysis.py --results-dir results --output-dir results/analysis

# Custom prices file
python scripts/result_analysis/run_analysis.py --prices scripts/result_analysis/prices.yaml

# Skip plots or JSON
python scripts/result_analysis/run_analysis.py --no-plots --no-json
# run experiment for lambda function
python3 scripts/client_lambda.py --function-name project-mlp --num-requests 1000 --data data/cancer_test_inputs.npy --output results/lambda_mlp_cancer_n1000_latencies.csv --region us-east-1  

# gather cloudwatch metrics for lambda function
python3 scripts/fetch_lambda_metrics.py --function-name project-mlp --region us-east-1 --start "2026-02-22 14:35:00" --end "2026-02-22 15:40:00" --output results/lambda_mlp_cancer_metrics.json
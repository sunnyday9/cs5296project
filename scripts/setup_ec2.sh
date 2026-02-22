# environment setup
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
git clone https://github.com/sunnyday9/cs5296project.git
cd cs5296project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# run server
python3 ec2/server.py --host 0.0.0.0 --port 5000 --model-type xgb --dataset adult

# run resource logger
python3 scripts/resource_logger.py --interval 1 --output results/resource_ec2_xgb_n100.csv

# run client
python3 scripts/client_ec2.py --url http://<推理服务器IP>:5000 --num-requests 100 --data data/test_input.npy --output results/ec2_latencies.csv
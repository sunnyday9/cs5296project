set -e
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
git clone https://github.com/sunnyday9/cs5296project.git
cd cs5296project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment setup complete."

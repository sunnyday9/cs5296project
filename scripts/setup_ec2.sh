set -e
REPO_URL="https://github.com/sunnyday9/cs5296project.git"
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
git clone "$REPO_URL"
cd cs5296project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment setup complete."

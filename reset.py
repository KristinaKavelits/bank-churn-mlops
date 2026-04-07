import shutil
import os
import subprocess
from pathlib import Path

def run_cmd(cmd):
    print(f"▶️ Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

print("\n🧹 1. Cleaning up old artifacts (MLflow, Reports, CSVs)...")
shutil.rmtree("mlruns", ignore_errors=True)
shutil.rmtree("reports", ignore_errors=True)
os.makedirs("reports", exist_ok=True)

for f in Path("data").glob("*.csv"):
    if f.name != "bank_customer_churn.csv":
        f.unlink()

print("\n🌱 2. Recreating data splits from scratch...")
run_cmd("./venv/bin/python src/ingestion.py")
run_cmd("./venv/bin/python data/create_drift.py")
run_cmd("./venv/bin/python data/create_target_drift.py")
run_cmd("./venv/bin/python data/prepare_test_batches.py")

print("\n🤖 3. Training initial baseline model (Day 0)...")
run_cmd("./venv/bin/python src/train.py")

print("\n✨ System reset complete! You are back to Day 0. ✨\n")

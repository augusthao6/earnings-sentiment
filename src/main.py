"""Main script to run the entire pipeline."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_command(command: str, description: str):
    """Run a command and print description."""
    print(f"\n=== {description} ===")
    result = subprocess.run(command, shell=True, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    else:
        print(result.stdout)

def main():
    """Run the full pipeline."""
    # 1. Prepare training data
    run_command("python src/prepare_training_data.py", "Preparing training data")

    # 2. Fine-tune FinBERT
    run_command("python src/fine_tune_finbert.py", "Fine-tuning FinBERT")

    # 3. Train fusion model
    run_command("python src/fusion_model.py", "Training fusion model")

    # 4. Run backtest
    run_command("python src/backtester.py", "Running backtest")

    print("\n=== Pipeline Complete ===")
    print("Check results in data/processed/ and models/")

if __name__ == "__main__":
    main()
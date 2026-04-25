"""Run the full pipeline: score → prepare → fine-tune → fusion → options → backtest."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(command: list[str], description: str) -> None:
    print(f"\n{'='*60}\n{description}\n{'='*60}")
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\nFailed at step: {description}")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full earnings-sentiment pipeline.")
    parser.add_argument(
        "--lm-only", action="store_true",
        help="Skip FinBERT scoring (fast, CPU-friendly). Pass this flag when not on GPU.",
    )
    parser.add_argument(
        "--skip-finetune", action="store_true",
        help="Skip FinBERT fine-tuning (use this on CPU — fine-tuning needs a GPU).",
    )
    args = parser.parse_args()

    py = sys.executable

    # 1. Score raw transcripts with LM dict (+ FinBERT if on GPU)
    score_cmd = [py, "src/score_transcripts.py"]
    if args.lm_only:
        score_cmd.append("--lm-only")
    run(score_cmd, "Step 1: Score transcripts (LM dictionary" + (" only)" if args.lm_only else " + FinBERT)"))

    # 2. Build training_data.csv: returns, EPS surprises, QoQ features, labels
    run([py, "src/prepare_training_data.py"], "Step 2: Prepare training data")

    # 3. Fine-tune FinBERT on Q&A sections (skip on CPU)
    if args.skip_finetune:
        print("\nSkipping FinBERT fine-tuning (--skip-finetune). Run on a GPU machine later.")
    else:
        run([py, "src/fine_tune_finbert.py"], "Step 3: Fine-tune FinBERT")

    # 4. Train Random Forest fusion model
    run([py, "src/fusion_model.py"], "Step 4: Train fusion model")

    # 5. Collect current options data (implied moves for upcoming earnings)
    run([py, "src/options_collection.py"], "Step 5: Collect options data")

    # 6. Backtest straddle strategy
    run([py, "src/backtester.py"], "Step 6: Backtest")

    print("\n" + "="*60)
    print("Pipeline complete.")
    print("  Training data : data/processed/training_data.csv")
    print("  Backtest results: data/processed/backtest_results.csv")
    print("  Options data  : data/processed/options_data.csv")
    print("  Fusion model  : models/fusion_model.pkl")
    if not args.skip_finetune:
        print("  Fine-tuned BERT: models/finbert_finetuned/")
    print("="*60)


if __name__ == "__main__":
    main()

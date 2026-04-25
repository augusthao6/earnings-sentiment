"""Backtest straddle strategy based on model predictions."""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def load_training_data() -> pd.DataFrame:
    """Load training data with predictions."""
    path = PROCESSED_DIR / "training_data.csv"
    return pd.read_csv(path)


def simulate_straddle_pnl(signal: int, realized_move: float, implied_move: float, premium: float = 0.05) -> float:
    """Simulate P&L for a straddle position."""
    if signal == 1:  # Buy straddle
        if realized_move > implied_move:
            # Profit: move beyond implied - premium
            pnl = (realized_move - implied_move) - premium
        else:
            # Loss: premium
            pnl = -premium
    else:  # No position
        pnl = 0
    return pnl


def backtest_strategy():
    """Backtest the strategy."""
    df = load_training_data()
    # Assume signal is the label for simplicity, or load model predictions
    # For now, use label as signal
    df['signal'] = df['label']
    df['pnl'] = df.apply(
        lambda row: simulate_straddle_pnl(row['signal'], row['realized_move'], row['implied_move']),
        axis=1
    )

    # Calculate metrics
    total_pnl = df['pnl'].sum()
    win_rate = (df['pnl'] > 0).mean()
    avg_pnl = df['pnl'].mean()
    sharpe = df['pnl'].mean() / df['pnl'].std() * np.sqrt(252) if df['pnl'].std() > 0 else 0

    print("Backtest Results:")
    print(f"Total P&L: {total_pnl:.4f}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average P&L per trade: {avg_pnl:.4f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")

    # Compare to baseline: always buy
    df['baseline_pnl'] = df.apply(
        lambda row: simulate_straddle_pnl(1, row['realized_move'], row['implied_move']),
        axis=1
    )
    baseline_total = df['baseline_pnl'].sum()
    print(f"Baseline (always buy) P&L: {baseline_total:.4f}")

    # Save results
    results_path = PROCESSED_DIR / "backtest_results.csv"
    df.to_csv(results_path, index=False)
    logger.info(f"Backtest results saved to {results_path}")


if __name__ == "__main__":
    backtest_strategy()
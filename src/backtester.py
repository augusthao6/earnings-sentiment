"""Backtest straddle strategy based on model predictions."""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from fusion_model import prepare_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Realistic at-the-money straddle premium for large-cap stocks (~2% of spot)
STRADDLE_PREMIUM = 0.02


def load_training_data() -> pd.DataFrame:
    """Load training data with predictions."""
    path = PROCESSED_DIR / "training_data.csv"
    return pd.read_csv(path)


def load_fusion_model():
    """Load the saved fusion model."""
    path = MODELS_DIR / "fusion_model.pkl"
    return joblib.load(path)


def simulate_straddle_pnl(signal: int, realized_move: float, implied_move: float, premium: float = STRADDLE_PREMIUM) -> float:
    """Simulate P&L for a straddle position."""
    if signal == 1:  # Buy straddle
        if realized_move > implied_move:
            pnl = (realized_move - implied_move) - premium
        else:
            pnl = -premium
    else:  # No position
        pnl = 0
    return pnl


def backtest_strategy():
    """Backtest the strategy using fusion model predictions."""
    df = load_training_data()
    model = load_fusion_model()

    features, _ = prepare_features(df)
    df['signal'] = model.predict(features)

    logger.info(
        "Model signals — buy: %d, no trade: %d (out of %d events)",
        (df['signal'] == 1).sum(), (df['signal'] == 0).sum(), len(df),
    )

    df['pnl'] = df.apply(
        lambda row: simulate_straddle_pnl(row['signal'], row['realized_move'], row['implied_move']),
        axis=1
    )

    # Metrics
    trades = df[df['signal'] == 1]
    total_pnl = df['pnl'].sum()
    win_rate = (trades['pnl'] > 0).mean() if len(trades) else float('nan')
    avg_pnl = df['pnl'].mean()
    sharpe = df['pnl'].mean() / df['pnl'].std() * np.sqrt(252) if df['pnl'].std() > 0 else 0

    print("Backtest Results:")
    print(f"Total P&L: {total_pnl:.4f}")
    print(f"Trades taken: {len(trades)}/{len(df)}")
    print(f"Win Rate (on trades): {win_rate:.2%}")
    print(f"Average P&L per event: {avg_pnl:.4f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")

    # Baseline: always buy
    df['baseline_pnl'] = df.apply(
        lambda row: simulate_straddle_pnl(1, row['realized_move'], row['implied_move']),
        axis=1
    )
    baseline_total = df['baseline_pnl'].sum()
    print(f"Baseline (always buy) P&L: {baseline_total:.4f}")

    results_path = PROCESSED_DIR / "backtest_results.csv"
    df.to_csv(results_path, index=False)
    logger.info(f"Backtest results saved to {results_path}")


if __name__ == "__main__":
    backtest_strategy()

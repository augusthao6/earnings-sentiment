"""Backtest straddle strategy based on model predictions.

Signal labels (3-class):
  0 = sell straddle  (model predicts small realized move — collect premium)
  1 = do nothing
  2 = buy straddle   (model predicts large realized move — pay premium)
"""

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

# Round-trip transaction cost as fraction of spot (~0.5% bid-ask on ATM straddle)
PREMIUM = 0.005


def load_training_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "training_data.csv"
    return pd.read_csv(path)


def load_fusion_model():
    path = MODELS_DIR / "fusion_model.pkl"
    return joblib.load(path)


def simulate_straddle_pnl(
    signal: int, realized_move: float, implied_move: float, premium: float = PREMIUM
) -> float:
    """Simulate per-event P&L as a fraction of spot.

    signal=2 (buy straddle):
      Pay implied_move to enter. Profit if realized > implied, net of costs.
    signal=0 (sell straddle):
      Collect implied_move. Profit if realized < implied, net of costs.
      Loss is uncapped if realized >> implied.
    signal=1 (do nothing): P&L = 0
    """
    if signal == 2:
        return (realized_move - implied_move) - premium if realized_move > implied_move else -premium
    if signal == 0:
        return (implied_move - realized_move) - premium
    return 0.0


def _annualized_sharpe(pnl_series: pd.Series, day0_dates: pd.Series) -> float:
    """Annualized Sharpe using actual event frequency (not sqrt(252))."""
    if pnl_series.std() == 0:
        return 0.0
    dates = pd.to_datetime(day0_dates)
    years = max((dates.max() - dates.min()).days / 365.25, 1.0)
    events_per_year = len(pnl_series) / years
    return pnl_series.mean() / pnl_series.std() * np.sqrt(events_per_year)


def backtest_strategy():
    df = load_training_data()
    model = load_fusion_model()

    features, _ = prepare_features(df)
    df["signal"] = model.predict(features)

    n_buy  = (df["signal"] == 2).sum()
    n_sell = (df["signal"] == 0).sum()
    n_hold = (df["signal"] == 1).sum()
    logger.info(
        "Model signals — buy: %d, sell: %d, hold: %d (out of %d events)",
        n_buy, n_sell, n_hold, len(df),
    )

    df["pnl"] = df.apply(
        lambda r: simulate_straddle_pnl(r["signal"], r["realized_move"], r["implied_move"]),
        axis=1,
    )

    trades = df[df["signal"] != 1]
    buy_trades = df[df["signal"] == 2]
    sell_trades = df[df["signal"] == 0]

    total_pnl = df["pnl"].sum()
    win_rate = (trades["pnl"] > 0).mean() if len(trades) else float("nan")
    avg_pnl = df["pnl"].mean()
    sharpe = _annualized_sharpe(df["pnl"], df["day_0_date"])

    print("\nBacktest Results:")
    print(f"  Total P&L:              {total_pnl:.4f}")
    print(f"  Active trades:          {len(trades)}/{len(df)} ({n_buy} buy, {n_sell} sell)")
    print(f"  Win Rate (trades only): {win_rate:.2%}")
    print(f"  Avg P&L per event:      {avg_pnl:.4f}")
    print(f"  Annualized Sharpe:      {sharpe:.2f}")

    # Baselines
    df["baseline_buy_pnl"] = df.apply(
        lambda r: simulate_straddle_pnl(2, r["realized_move"], r["implied_move"]), axis=1
    )
    df["baseline_sell_pnl"] = df.apply(
        lambda r: simulate_straddle_pnl(0, r["realized_move"], r["implied_move"]), axis=1
    )
    print(f"\n  Baseline always-buy P&L:  {df['baseline_buy_pnl'].sum():.4f}")
    print(f"  Baseline always-sell P&L: {df['baseline_sell_pnl'].sum():.4f}")

    results_path = PROCESSED_DIR / "backtest_results.csv"
    df.to_csv(results_path, index=False)
    logger.info("Backtest results saved to %s", results_path)


if __name__ == "__main__":
    backtest_strategy()

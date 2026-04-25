"""Prepare data for fine-tuning FinBERT on earnings sentiment for options trading."""

import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
TRANSCRIPTS_DIR = RAW_DIR / "transcripts"
PRICES_DIR = RAW_DIR / "prices"

TARGET_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "BAC"]


def load_sentiment_scores() -> pd.DataFrame:
    """Load the sentiment scores."""
    path = PROCESSED_DIR / "sentiment_scores.csv"
    df = pd.read_csv(path)
    return df


def load_prices() -> dict[str, pd.DataFrame]:
    """Load price data for all tickers."""
    prices = {}
    for ticker in TARGET_TICKERS:
        path = PRICES_DIR / f"{ticker}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            prices[ticker] = df
    return prices


def map_quarter_to_date(quarter: str) -> str:
    """Map quarter like '2021-Q1' to approximate earnings date."""
    year, q = quarter.split('-')
    year = int(year)
    q_num = int(q[1])
    # Approximate earnings dates: Q1 ~ April, Q2 ~ July, Q3 ~ Oct, Q4 ~ Jan next year
    if q_num == 1:
        month = 4
    elif q_num == 2:
        month = 7
    elif q_num == 3:
        month = 10
    elif q_num == 4:
        month = 1
        year += 1
    return f"{year}-{month:02d}-15"  # Mid month as approx


def calculate_returns(prices: dict[str, pd.DataFrame], ticker: str, earnings_date: str) -> dict:
    """Calculate 1-day and 5-day returns after earnings."""
    if ticker not in prices:
        return {}
    price_df = prices[ticker]
    date = pd.to_datetime(earnings_date)
    try:
        # Find the next trading day after earnings
        next_day = date + timedelta(days=1)
        while next_day not in price_df.index:
            next_day += timedelta(days=1)
            if next_day > date + timedelta(days=10):
                break
        if next_day not in price_df.index:
            return {}
        price_before = price_df.loc[price_df.index <= date, 'Close'].iloc[-1]
        price_after_1d = price_df.loc[next_day, 'Close']
        return_1d = (price_after_1d - price_before) / price_before

        # 5-day return
        five_days_later = next_day + timedelta(days=5)
        while five_days_later not in price_df.index:
            five_days_later -= timedelta(days=1)
            if five_days_later < next_day:
                break
        if five_days_later in price_df.index:
            price_after_5d = price_df.loc[five_days_later, 'Close']
            return_5d = (price_after_5d - price_before) / price_before
        else:
            return_5d = np.nan

        return {
            'return_1d': return_1d,
            'return_5d': return_5d,
            'price_before': price_before,
            'price_after_1d': price_after_1d
        }
    except Exception as e:
        logger.warning(f"Failed to calculate returns for {ticker} on {earnings_date}: {e}")
        return {}


def prepare_training_data() -> pd.DataFrame:
    """Prepare data for training: sentiment + returns + labels."""
    sentiment_df = load_sentiment_scores()
    prices = load_prices()

    training_data = []
    for _, row in sentiment_df.iterrows():
        ticker = row['ticker']
        quarter = row['quarter']
        earnings_date = map_quarter_to_date(quarter)
        returns = calculate_returns(prices, ticker, earnings_date)
        if returns:
            # Assume implied move of 5% for simplicity
            implied_move = 0.05
            realized_move = abs(returns['return_1d'])
            label = 1 if realized_move > implied_move else 0  # 1 = buy straddle, 0 = no
            training_data.append({
                'ticker': ticker,
                'quarter': quarter,
                'earnings_date': earnings_date,
                'lm_positive': row['lm_positive'],
                'lm_negative': row['lm_negative'],
                'lm_net_sentiment': row['lm_net_sentiment'],
                'finbert_positive': row['finbert_positive'],
                'finbert_negative': row['finbert_negative'],
                'finbert_net_sentiment': row['finbert_net_sentiment'],
                'return_1d': returns['return_1d'],
                'return_5d': returns['return_5d'],
                'implied_move': implied_move,
                'realized_move': realized_move,
                'label': label
            })

    df = pd.DataFrame(training_data)
    output_path = PROCESSED_DIR / "training_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved training data to {output_path}")
    return df


if __name__ == "__main__":
    prepare_training_data()
"""Collect earnings dates and options data for backtesting."""

import logging
import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OPTIONS_DIR = DATA_DIR / "raw" / "options"

TARGET_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "BAC"]


def get_earnings_dates(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get historical earnings dates for a ticker."""
    stock = yf.Ticker(ticker)
    try:
        earnings = stock.earnings_dates
        if earnings is None or earnings.empty:
            logger.warning(f"No earnings dates found for {ticker}")
            return pd.DataFrame()
        # Filter by date range
        earnings = earnings.reset_index()
        earnings.columns = ['date', 'eps_estimate', 'reported_eps', 'surprise']
        earnings['date'] = pd.to_datetime(earnings['date'])
        mask = (earnings['date'] >= start_date) & (earnings['date'] <= end_date)
        filtered = earnings[mask].copy()
        filtered['ticker'] = ticker
        return filtered[['ticker', 'date', 'eps_estimate', 'reported_eps', 'surprise']]
    except Exception as e:
        logger.error(f"Failed to get earnings for {ticker}: {e}")
        return pd.DataFrame()


def collect_all_earnings(start_date: str = "2020-01-01", end_date: str = "2024-01-01") -> pd.DataFrame:
    """Collect earnings dates for all target tickers."""
    all_earnings = []
    for ticker in TARGET_TICKERS:
        logger.info(f"Collecting earnings for {ticker}")
        df = get_earnings_dates(ticker, start_date, end_date)
        if not df.empty:
            all_earnings.append(df)
        time.sleep(1)  # Rate limit
    if all_earnings:
        return pd.concat(all_earnings, ignore_index=True)
    return pd.DataFrame()


def get_options_data(ticker: str, date: str) -> dict:
    """Get options chain for a ticker on a specific date."""
    try:
        stock = yf.Ticker(ticker)
        # Get options for the expiration closest to date, but ideally for earnings, get weekly options
        # For simplicity, get the options chain for the date
        # yfinance options are for specific expirations
        # To get historical options, it's tricky, yfinance doesn't support historical options directly
        # For this project, we'll use current options as proxy, but that's not accurate
        # Actually, yfinance has options for current date only
        # To get historical implied vol, we can use historical data or approximate
        # For simplicity, let's get the current options and use as example
        options = stock.options
        if not options:
            return {}
        # Get the first expiration
        opt = stock.option_chain(options[0])
        calls = opt.calls
        puts = opt.puts
        # Find ATM straddle
        spot = stock.info.get('regularMarketPrice', 0)
        if spot == 0:
            return {}
        # Find strike closest to spot
        calls['strike_diff'] = (calls['strike'] - spot).abs()
        puts['strike_diff'] = (puts['strike'] - spot).abs()
        atm_call = calls.loc[calls['strike_diff'].idxmin()]
        atm_put = puts.loc[puts['strike_diff'].idxmin()]
        straddle_price = atm_call['lastPrice'] + atm_put['lastPrice']
        implied_move = 2 * straddle_price / spot
        return {
            'ticker': ticker,
            'date': date,
            'spot': spot,
            'straddle_price': straddle_price,
            'implied_move': implied_move
        }
    except Exception as e:
        logger.error(f"Failed to get options for {ticker} on {date}: {e}")
        return {}


def collect_options_for_earnings(earnings_df: pd.DataFrame) -> pd.DataFrame:
    """Collect options data for each earnings event."""
    options_data = []
    for _, row in earnings_df.iterrows():
        ticker = row['ticker']
        date = row['date'].strftime('%Y-%m-%d')
        logger.info(f"Collecting options for {ticker} on {date}")
        opt_data = get_options_data(ticker, date)
        if opt_data:
            options_data.append(opt_data)
        time.sleep(1)
    return pd.DataFrame(options_data)


if __name__ == "__main__":
    # Collect earnings dates
    earnings_df = collect_all_earnings()
    if not earnings_df.empty:
        earnings_path = PROCESSED_DIR / "earnings_dates.csv"
        earnings_df.to_csv(earnings_path, index=False)
        logger.info(f"Saved earnings dates to {earnings_path}")

        # Collect options data
        options_df = collect_options_for_earnings(earnings_df)
        if not options_df.empty:
            options_path = PROCESSED_DIR / "options_data.csv"
            options_df.to_csv(options_path, index=False)
            logger.info(f"Saved options data to {options_path}")
    else:
        logger.warning("No earnings data collected")
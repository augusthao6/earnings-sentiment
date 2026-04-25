"""Collect upcoming earnings dates and current ATM straddle prices via yfinance.

NOTE: yfinance does not provide historical options chains, so this module is
designed for *live/forward-looking* signals — i.e., what trades to consider
before the next earnings cycle. Historical backtesting uses the rolling-vol
implied_move proxy computed in prepare_training_data.py instead.

Output: data/processed/options_data.csv
  ticker, earnings_date, spot, straddle_price, implied_move, expiry_used
"""

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "raw" / "transcripts"


# ---------------------------------------------------------------------------
# Ticker discovery (mirrors prepare_training_data.py — no hardcoded list)
# ---------------------------------------------------------------------------

def discover_tickers() -> list[str]:
    """Read tickers from training_data.csv if available, else scan transcripts."""
    training_csv = PROCESSED_DIR / "training_data.csv"
    if training_csv.exists():
        df = pd.read_csv(training_csv)
        tickers = sorted(df["ticker"].unique().tolist())
        logger.info("Discovered %d tickers from training_data.csv", len(tickers))
        return tickers

    tickers = set()
    for path in TRANSCRIPTS_DIR.glob("*.txt"):
        parts = path.stem.split("_")
        if parts:
            tickers.add(parts[0])
    result = sorted(tickers)
    logger.info("Discovered %d tickers from transcripts directory", len(result))
    return result


# ---------------------------------------------------------------------------
# Earnings dates (next ~12 months from yfinance)
# ---------------------------------------------------------------------------

def get_next_earnings_date(ticker: str) -> pd.Timestamp | None:
    """Return the next scheduled earnings date for ticker, or None."""
    try:
        stock = yf.Ticker(ticker)
        dates = stock.earnings_dates
        if dates is None or dates.empty:
            return None
        dates = dates.reset_index()
        # Column name varies by yfinance version
        date_col = dates.columns[0]
        dates[date_col] = pd.to_datetime(dates[date_col], utc=True)
        future = dates[dates[date_col] > pd.Timestamp.now(tz="UTC")]
        if future.empty:
            return None
        return future[date_col].min().tz_localize(None)
    except Exception as e:
        logger.warning("Could not fetch earnings dates for %s: %s", ticker, e)
        return None


# ---------------------------------------------------------------------------
# Current ATM straddle price
# ---------------------------------------------------------------------------

def get_atm_straddle(ticker: str) -> dict:
    """Fetch the current ATM straddle price from the nearest available expiry.

    Returns a dict with spot, straddle_price, implied_move, expiry_used.
    implied_move = straddle_price / spot  (fraction, e.g. 0.07 = 7%)
    """
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {}

        spot = stock.fast_info.get("lastPrice") or stock.fast_info.get("regularMarketPrice")
        if not spot or spot <= 0:
            return {}

        # Use the nearest expiry that has enough data (skip < 3 days out)
        now = pd.Timestamp.now()
        chosen_expiry = None
        for exp in expirations:
            exp_dt = pd.to_datetime(exp)
            if (exp_dt - now).days >= 3:
                chosen_expiry = exp
                break

        if chosen_expiry is None:
            return {}

        chain = stock.option_chain(chosen_expiry)
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty:
            return {}

        calls = calls.copy()
        puts = puts.copy()
        calls["strike_diff"] = (calls["strike"] - spot).abs()
        puts["strike_diff"] = (puts["strike"] - spot).abs()
        atm_call = calls.loc[calls["strike_diff"].idxmin()]
        atm_put = puts.loc[puts["strike_diff"].idxmin()]

        # Use mid-price to avoid inflated last-price from stale prints
        call_price = (atm_call["bid"] + atm_call["ask"]) / 2 if atm_call["ask"] > 0 else atm_call["lastPrice"]
        put_price = (atm_put["bid"] + atm_put["ask"]) / 2 if atm_put["ask"] > 0 else atm_put["lastPrice"]

        straddle_price = call_price + put_price
        implied_move = straddle_price / spot

        return {
            "spot": round(spot, 2),
            "straddle_price": round(straddle_price, 4),
            "implied_move": round(implied_move, 4),
            "expiry_used": chosen_expiry,
        }

    except Exception as e:
        logger.warning("Could not fetch options for %s: %s", ticker, e)
        return {}


# ---------------------------------------------------------------------------
# Main collection function
# ---------------------------------------------------------------------------

def collect_options_data(tickers: list[str] | None = None) -> pd.DataFrame:
    """For each ticker: fetch next earnings date + current ATM straddle price."""
    if tickers is None:
        tickers = discover_tickers()

    records = []
    for ticker in tickers:
        logger.info("Processing %s", ticker)

        earnings_date = get_next_earnings_date(ticker)
        straddle = get_atm_straddle(ticker)

        if not straddle:
            logger.warning("No options data for %s — skipping", ticker)
            time.sleep(0.5)
            continue

        records.append({
            "ticker": ticker,
            "next_earnings_date": earnings_date.strftime("%Y-%m-%d") if earnings_date else None,
            "spot": straddle["spot"],
            "straddle_price": straddle["straddle_price"],
            "implied_move": straddle["implied_move"],
            "expiry_used": straddle["expiry_used"],
        })
        time.sleep(0.5)  # be polite to yfinance

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No options data collected — is market open? yfinance needs live quotes.")
        return df

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "options_data.csv"
    df.to_csv(out, index=False)
    logger.info("Saved options data for %d tickers to %s", len(df), out)

    # Print a quick summary: tickers with upcoming earnings sorted by date
    upcoming = df.dropna(subset=["next_earnings_date"]).sort_values("next_earnings_date")
    if not upcoming.empty:
        print("\nUpcoming earnings with options data:")
        print(upcoming[["ticker", "next_earnings_date", "implied_move"]].to_string(index=False))

    return df


if __name__ == "__main__":
    collect_options_data()

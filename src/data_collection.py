"""Load earnings call transcripts from Kaggle pickle or FMP API, and stock prices via yfinance."""

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Paths ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_PICKLE = PROJECT_ROOT / "data" / "raw" / "kaggle_source" / "motley-fool-data.pkl"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "raw" / "transcripts"
PRICES_DIR = PROJECT_ROOT / "data" / "raw" / "prices"
EPS_DIR = PROJECT_ROOT / "data" / "raw" / "eps_surprises"
FMP_DATES_CSV = PROJECT_ROOT / "data" / "raw" / "fmp_transcript_dates.csv"

# --- Kaggle pipeline config ---

TARGET_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "BAC"]
MIN_QUARTER = "2021-Q1"
MAX_QUARTER = "2023-Q2"
MIN_TRANSCRIPTS = 7
PRICE_START = "2020-12-01"
PRICE_END = "2023-09-30"

# --- FMP config ---

FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
FMP_RATE_LIMIT_SLEEP = 0.5  # seconds between requests (free tier: 250/day)


# ---------------------------------------------------------------------------
# FMP helpers
# ---------------------------------------------------------------------------

def _fmp_get(endpoint: str, params: dict) -> dict | list | None:
    """Make a single FMP API request; return parsed JSON or None on error."""
    if not FMP_API_KEY:
        raise ValueError(
            "FMP_API_KEY is not set. Get a free key at financialmodelingprep.com "
            "and set it with:  import os; os.environ['FMP_API_KEY'] = 'your_key'"
        )
    params["apikey"] = FMP_API_KEY
    url = f"{FMP_BASE_URL}/{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "Error Message" in data:
            logger.error("FMP API error: %s", data["Error Message"])
            return None
        return data
    except Exception as e:
        logger.warning("FMP request failed (%s): %s", endpoint, e)
        return None


def _estimate_api_calls(tickers: list[str], min_year: int, max_year: int,
                        fetch_prices: bool, fetch_eps: bool) -> int:
    n_quarters = (max_year - min_year + 1) * 4
    transcript_calls = len(tickers) * n_quarters
    price_calls = len(tickers) if fetch_prices else 0
    eps_calls = len(tickers) if fetch_eps else 0
    return transcript_calls + price_calls + eps_calls


# ---------------------------------------------------------------------------
# FMP: transcripts
# ---------------------------------------------------------------------------

def fetch_fmp_transcripts(
    tickers: list[str],
    min_year: int,
    max_year: int,
    output_dir: Path,
) -> list[dict]:
    """Download transcripts from FMP and save as {TICKER}_{YEAR}_Q{N}.txt.

    Also returns a list of metadata dicts (ticker, quarter, call_datetime,
    is_after_hours) for every transcript fetched or found on disk so that
    prepare_training_data.py knows the actual earnings call time without
    needing the Kaggle pickle.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_records = []

    for ticker in tickers:
        for year in range(min_year, max_year + 1):
            for quarter in range(1, 5):
                quarter_str = f"{year}-Q{quarter}"
                filename = f"{ticker}_{year}_Q{quarter}.txt"
                filepath = output_dir / filename

                if filepath.exists():
                    logger.info("Skipping %s (already on disk)", filename)
                    # Still record metadata if we already have it saved
                    continue

                logger.info("Fetching %s Q%d %d", ticker, quarter, year)
                data = _fmp_get(
                    f"earning_call_transcript/{ticker}",
                    {"quarter": quarter, "year": year},
                )
                time.sleep(FMP_RATE_LIMIT_SLEEP)

                if not data:
                    continue
                entry = data[0] if isinstance(data, list) and data else data
                content = entry.get("content", "")
                if not content:
                    logger.warning("Empty transcript for %s Q%d %d", ticker, quarter, year)
                    continue

                filepath.write_text(content, encoding="utf-8")

                # Parse the date FMP provides (e.g. "2022-01-28 17:00:00")
                raw_date = entry.get("date", "")
                try:
                    call_dt = pd.to_datetime(raw_date)
                    is_after_hours = call_dt.hour >= 16
                    metadata_records.append({
                        "ticker": ticker,
                        "quarter": quarter_str,
                        "call_datetime": call_dt,
                        "is_after_hours": is_after_hours,
                    })
                except Exception:
                    logger.warning("Could not parse date '%s' for %s %s", raw_date, ticker, quarter_str)

    return metadata_records


def _save_fmp_dates(new_records: list[dict]) -> None:
    """Append new metadata records to the FMP dates CSV (deduplicating on ticker+quarter)."""
    if not new_records:
        return
    new_df = pd.DataFrame(new_records)
    if FMP_DATES_CSV.exists():
        existing = pd.read_csv(FMP_DATES_CSV, parse_dates=["call_datetime"])
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=["ticker", "quarter"], keep="last")
    FMP_DATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(FMP_DATES_CSV, index=False)
    logger.info("Saved %d total date records to %s", len(combined), FMP_DATES_CSV)


# ---------------------------------------------------------------------------
# FMP: EPS surprises  (addresses Bias 3: controls for headline beat/miss)
# ---------------------------------------------------------------------------

def fetch_all_eps_surprises(tickers: list[str], output_dir: Path) -> None:
    """Download historical EPS beat/miss data for each ticker (1 API call per ticker).

    Saves {TICKER}.csv with columns: date, actualEarningResult, estimatedEarning,
    eps_surprise_pct, eps_beat.

    This is the key control variable for earnings call analysis. The stock moves
    primarily because of the EPS beat/miss; the transcript's residual signal is
    tone and forward guidance *after* controlling for the headline number.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        filepath = output_dir / f"{ticker}.csv"
        if filepath.exists():
            logger.info("Skipping EPS surprises for %s (already on disk)", ticker)
            continue

        logger.info("Fetching EPS surprises for %s", ticker)
        data = _fmp_get(f"earnings-surprises/{ticker}", {})
        time.sleep(FMP_RATE_LIMIT_SLEEP)
        if not data:
            logger.warning("No EPS surprise data for %s", ticker)
            continue

        df = pd.DataFrame(data)
        if df.empty or "actualEarningResult" not in df.columns:
            continue

        df["date"] = pd.to_datetime(df["date"])
        # eps_surprise_pct: how much did actual EPS differ from estimate (as a fraction)?
        df["eps_surprise_pct"] = (
            (df["actualEarningResult"] - df["estimatedEarning"])
            / df["estimatedEarning"].abs().replace(0, float("nan"))
        )
        df["eps_beat"] = (df["actualEarningResult"] >= df["estimatedEarning"]).astype(int)
        df[["date", "actualEarningResult", "estimatedEarning",
            "eps_surprise_pct", "eps_beat"]].to_csv(filepath, index=False)
        logger.info("Saved %d EPS surprise records for %s", len(df), ticker)


# ---------------------------------------------------------------------------
# FMP: prices
# ---------------------------------------------------------------------------

def fetch_fmp_prices(
    tickers: list[str],
    start: str,
    end: str,
    output_dir: Path,
) -> tuple[int, list[str]]:
    """Download daily adjusted close prices from FMP; fall back to yfinance on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    success, failed = 0, []

    for ticker in tickers:
        filepath = output_dir / f"{ticker}.csv"
        if filepath.exists():
            logger.info("Skipping prices for %s (already on disk)", ticker)
            success += 1
            continue

        data = _fmp_get(f"historical-price-full/{ticker}", {"from": start, "to": end})
        if data and "historical" in data:
            price_df = pd.DataFrame(data["historical"])
            price_df["date"] = pd.to_datetime(price_df["date"])
            price_df = price_df.set_index("date").sort_index()
            price_df = price_df.rename(columns={"adjClose": "Close"})
            price_df.to_csv(filepath)
            logger.info("Saved %d price rows for %s (FMP)", len(price_df), ticker)
            success += 1
            time.sleep(FMP_RATE_LIMIT_SLEEP)
            continue

        # Fall back to yfinance
        logger.warning("FMP prices failed for %s — falling back to yfinance", ticker)
        try:
            price_df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if price_df.empty:
                raise ValueError(f"Empty yfinance result for {ticker}")
            if isinstance(price_df.columns, pd.MultiIndex):
                price_df.columns = price_df.columns.get_level_values(0)
            price_df.to_csv(filepath)
            logger.info("Saved %d price rows for %s (yfinance)", len(price_df), ticker)
            success += 1
        except Exception as e:
            logger.error("Permanently failed prices for %s: %s", ticker, e)
            failed.append(ticker)

    return success, failed


# ---------------------------------------------------------------------------
# Kaggle source (original pipeline, unchanged)
# ---------------------------------------------------------------------------

def load_raw_data(pickle_path: Path) -> pd.DataFrame:
    logger.info("Loading raw data from %s", pickle_path)
    return pd.read_pickle(pickle_path)


def filter_by_tickers_and_dates(df, tickers, min_q, max_q):
    mask = df["ticker"].isin(tickers) & (df["q"] >= min_q) & (df["q"] <= max_q)
    filtered = df[mask].copy()
    logger.info("Filtered to %d rows for %d tickers", len(filtered), len(tickers))
    return filtered


def deduplicate(df):
    df["transcript_len"] = df["transcript"].str.len()
    deduped = (
        df.sort_values("transcript_len", ascending=False)
        .drop_duplicates(subset=["ticker", "q"], keep="first")
        .drop(columns=["transcript_len"])
    )
    dropped = len(df) - len(deduped)
    if dropped:
        logger.info("Dropped %d duplicate rows", dropped)
    return deduped


def enforce_min_transcripts(df, min_count):
    counts = df.groupby("ticker").size()
    keep = counts[counts >= min_count].index.tolist()
    drop = counts[counts < min_count].index.tolist()
    for ticker in drop:
        logger.warning("Dropping %s — only %d transcripts", ticker, counts[ticker])
    return df[df["ticker"].isin(keep)].copy(), drop


def save_transcripts(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for _, row in df.iterrows():
        year, quarter_str = row["q"].split("-")
        filename = f"{row['ticker']}_{year}_{quarter_str}.txt"
        (output_dir / filename).write_text(row["transcript"], encoding="utf-8")
        saved += 1
    logger.info("Saved %d transcript files to %s", saved, output_dir)
    return saved


def download_prices(tickers, start, end, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    success, failed = 0, []
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(2)
        try:
            price_df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if price_df.empty:
                raise ValueError(f"Empty result for {ticker}")
            if isinstance(price_df.columns, pd.MultiIndex):
                price_df.columns = price_df.columns.get_level_values(0)
            (output_dir / f"{ticker}.csv").write_text(
                price_df.to_csv(), encoding="utf-8"
            )
            success += 1
        except Exception as e:
            logger.error("Failed prices for %s: %s", ticker, e)
            failed.append(ticker)
    return success, failed


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def main_kaggle() -> None:
    """Original pipeline: load from committed Kaggle pickle."""
    raw_df = load_raw_data(KAGGLE_PICKLE)
    filtered = filter_by_tickers_and_dates(raw_df, TARGET_TICKERS, MIN_QUARTER, MAX_QUARTER)
    deduped = deduplicate(filtered)
    final_df, _ = enforce_min_transcripts(deduped, MIN_TRANSCRIPTS)
    final_tickers = sorted(final_df["ticker"].unique().tolist())
    save_transcripts(final_df, TRANSCRIPTS_DIR)
    download_prices(final_tickers, PRICE_START, PRICE_END, PRICES_DIR)


def main_fmp(
    tickers: list[str] | None = None,
    min_year: int = 2023,
    max_year: int = 2026,
    price_start: str = "2022-10-01",
    price_end: str = "2026-12-31",
) -> None:
    """Pull transcripts, prices, and EPS surprises from FMP for the given date range.

    2023-2026 fits comfortably in one session (~225 actual API calls after
    skipping empty/future quarters). Files already on disk are always skipped,
    so this is safe to re-run if it's interrupted.

    Example:
        import os; os.environ["FMP_API_KEY"] = "your_key"
        from src.data_collection import main_fmp
        main_fmp(
            tickers=["AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL",
                     "JPM", "BAC", "GS", "XOM", "CVX", "PFE", "UNH", "AMD"],
        )
    """
    if tickers is None:
        tickers = TARGET_TICKERS

    est = _estimate_api_calls(tickers, min_year, max_year, fetch_prices=True, fetch_eps=True)
    logger.info(
        "tickers=%d | years=%d-%d | estimated API calls=%d "
        "(actual will be lower — empty/future quarters are skipped)",
        len(tickers), min_year, max_year, est,
    )
    if est > 250:
        logger.warning(
            "Upper-bound estimate exceeds 250. In practice future quarters return "
            "immediately with no data, so actual calls stay under the limit. "
            "If you do hit it, re-run tomorrow — already-saved files are skipped."
        )

    logger.info("=== Fetching prices ===")
    fetch_fmp_prices(tickers, price_start, price_end, PRICES_DIR)

    logger.info("=== Fetching transcripts %d-%d ===", min_year, max_year)
    new_metadata = fetch_fmp_transcripts(tickers, min_year, max_year, TRANSCRIPTS_DIR)
    _save_fmp_dates(new_metadata)
    logger.info("Fetched %d new transcript date records", len(new_metadata))

    logger.info("=== Fetching EPS surprises ===")
    fetch_all_eps_surprises(tickers, EPS_DIR)

    logger.info("Collection complete.")


if __name__ == "__main__":
    main_kaggle()

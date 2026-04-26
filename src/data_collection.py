"""Collect earnings call transcripts (Motley Fool), prices (yfinance), and EPS surprises (FMP)."""

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

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

# --- FMP config (used only for EPS surprises on free tier) ---

FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
FMP_RATE_LIMIT_SLEEP = 0.5

# --- Motley Fool config ---

# ---------------------------------------------------------------------------
# FMP helper (used only for EPS surprises)
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


# ---------------------------------------------------------------------------
# Prices: yfinance (free, no quota)
# ---------------------------------------------------------------------------

def fetch_fmp_prices(
    tickers: list[str],
    start: str,
    end: str,
    output_dir: Path,
) -> tuple[int, list[str]]:
    """Download daily adjusted close prices via yfinance.

    FMP historical price endpoints require a paid plan; yfinance is free
    and covers all tickers. Files already on disk are skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    success, failed = 0, []

    for ticker in tickers:
        filepath = output_dir / f"{ticker}.csv"
        if filepath.exists():
            logger.info("Skipping prices for %s (already on disk)", ticker)
            success += 1
            continue

        try:
            price_df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if price_df.empty:
                raise ValueError(f"Empty result for {ticker}")
            if isinstance(price_df.columns, pd.MultiIndex):
                price_df.columns = price_df.columns.get_level_values(0)
            price_df.to_csv(filepath)
            logger.info("Saved %d price rows for %s", len(price_df), ticker)
            success += 1
        except Exception as e:
            logger.error("Failed prices for %s: %s", ticker, e)
            failed.append(ticker)

    return success, failed


# ---------------------------------------------------------------------------
# SEC EDGAR earnings press release fetcher
# ---------------------------------------------------------------------------

def _edgar_get_cik_map() -> dict[str, str]:
    """Fetch ticker → zero-padded CIK mapping from SEC EDGAR."""
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": "earnings-sentiment-research/1.0 student@university.edu"},
        timeout=15,
    )
    resp.raise_for_status()
    return {
        entry["ticker"]: str(entry["cik_str"]).zfill(10)
        for entry in resp.json().values()
    }


def _edgar_download_exhibit(cik: str, accession: str) -> str:
    """Download the EX-99.1 press release text from an SEC 8-K filing.

    Uses the filing's JSON index to locate the exhibit, then extracts plain text.
    Returns empty string if the exhibit cannot be found or parsed.
    """
    headers = {"User-Agent": "earnings-sentiment-research/1.0 student@university.edu"}
    acc_nodash = accession.replace("-", "")
    cik_int = int(cik)

    # EDGAR provides a JSON filing index
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
        f"{acc_nodash}/{accession}-index.json"
    )
    try:
        resp = requests.get(index_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return ""
        docs = resp.json().get("documents", [])
    except Exception:
        return ""

    # Prefer EX-99.1 (earnings press release); fall back to first .htm document
    exhibit_url = None
    fallback_url = None
    for doc in docs:
        doc_type = doc.get("type", "")
        filename = doc.get("document", "")
        if not filename:
            continue
        full_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{filename}"
        )
        if doc_type in ("EX-99.1", "EX-99") or "99" in doc_type:
            exhibit_url = full_url
            break
        if filename.endswith(".htm") and fallback_url is None:
            fallback_url = full_url

    target = exhibit_url or fallback_url
    if not target:
        return ""

    try:
        time.sleep(0.1)
        resp = requests.get(target, headers=headers, timeout=15)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except Exception:
        return ""


def fetch_edgar_earnings_releases(
    tickers: list[str],
    min_year: int,
    max_year: int,
    output_dir: Path,
) -> list[dict]:
    """Fetch earnings press releases from SEC EDGAR 8-K Item 2.02 filings.

    Every US public company must file their quarterly earnings announcement as
    a Form 8-K Item 2.02 (Results of Operations and Financial Condition).
    The EX-99.1 exhibit is the full press release with management commentary
    and forward guidance — the text used for LM sentiment scoring.

    SEC EDGAR API is free, official, and has no bot-detection issues.
    Rate limit: 10 req/s — we stay well under that with 0.1 s sleeps.

    Returns metadata records compatible with _save_fmp_dates().
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "earnings-sentiment-research/1.0 student@university.edu"}
    metadata_records = []

    # Step 1: fetch ticker → CIK map once
    logger.info("Fetching ticker→CIK map from SEC EDGAR...")
    try:
        cik_map = _edgar_get_cik_map()
    except Exception as e:
        logger.error("Could not load CIK map: %s", e)
        return []

    # Step 2: get yfinance earnings dates so we can match 8-Ks to earnings events
    logger.info("Fetching earnings dates from yfinance...")
    yf_dates: dict[str, list[pd.Timestamp]] = {}
    for ticker in tickers:
        try:
            ed = yf.Ticker(ticker).earnings_dates
            if ed is not None and not ed.empty:
                ed = ed.reset_index()
                col = ed.columns[0]
                if ed[col].dt.tz is not None:
                    ed[col] = ed[col].dt.tz_localize(None)
                yf_dates[ticker] = ed[col].tolist()
        except Exception:
            yf_dates[ticker] = []

    start_ts = pd.Timestamp(f"{min_year}-01-01")
    end_ts = min(pd.Timestamp(f"{max_year}-12-31"), pd.Timestamp.now())

    for ticker in tickers:
        cik = cik_map.get(ticker)
        if not cik:
            logger.warning("CIK not found for %s — skipping", ticker)
            continue

        # Step 3: fetch company's recent filings from EDGAR
        try:
            time.sleep(0.1)
            resp = requests.get(
                f"https://data.sec.gov/submissions/CIK{cik}.json",
                headers=headers, timeout=15,
            )
            resp.raise_for_status()
            recent = resp.json().get("filings", {}).get("recent", {})
        except Exception as e:
            logger.warning("Could not fetch EDGAR submissions for %s: %s", ticker, e)
            continue

        forms   = recent.get("form", [])
        dates   = recent.get("filingDate", [])
        accs    = recent.get("accessionNumber", [])
        items   = recent.get("items", [])

        known_dates = yf_dates.get(ticker, [])

        for form, date_str, acc, item_str in zip(forms, dates, accs, items):
            if form != "8-K":
                continue
            # Item 2.02 = Results of Operations (earnings release)
            if "2.02" not in str(item_str):
                continue

            filing_date = pd.Timestamp(date_str)
            if not (start_ts <= filing_date <= end_ts):
                continue

            # Match to a known yfinance earnings date (within 7 days)
            call_date = filing_date
            if known_dates:
                deltas = [(abs((filing_date - d).days), d) for d in known_dates]
                best_delta, best_date = min(deltas, key=lambda x: x[0])
                if best_delta > 7:
                    continue  # not an earnings event we care about
                call_date = best_date

            q = (filing_date.month - 1) // 3 + 1
            year = filing_date.year
            quarter_str = f"{year}-Q{q}"
            filename = f"{ticker}_{year}_Q{q}.txt"
            filepath = output_dir / filename

            if filepath.exists():
                logger.info("Skipping %s (already on disk)", filename)
                continue

            logger.info("Fetching SEC 8-K: %s %s", ticker, quarter_str)
            text = _edgar_download_exhibit(cik, acc)
            if len(text) < 500:
                logger.warning("8-K exhibit too short for %s %s (%d chars)", ticker, quarter_str, len(text))
                continue

            filepath.write_text(text, encoding="utf-8")
            logger.info("Saved %s (%d chars)", filename, len(text))

            is_after_hours = call_date.hour >= 16 if call_date.hour != 0 else True
            metadata_records.append({
                "ticker": ticker,
                "quarter": quarter_str,
                "call_datetime": call_date,
                "is_after_hours": is_after_hours,
            })

    logger.info("SEC EDGAR: collected %d earnings releases", len(metadata_records))
    return metadata_records


# ---------------------------------------------------------------------------
# FMP: EPS surprises (free tier supports this endpoint)
# ---------------------------------------------------------------------------

def fetch_all_eps_surprises(tickers: list[str], output_dir: Path) -> None:
    """Download historical EPS beat/miss data for each ticker (1 API call per ticker).

    Saves {TICKER}.csv with columns: date, actualEarningResult, estimatedEarning,
    eps_surprise_pct, eps_beat.
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
        df["eps_surprise_pct"] = (
            (df["actualEarningResult"] - df["estimatedEarning"])
            / df["estimatedEarning"].abs().replace(0, float("nan"))
        )
        df["eps_beat"] = (df["actualEarningResult"] >= df["estimatedEarning"]).astype(int)
        df[["date", "actualEarningResult", "estimatedEarning",
            "eps_surprise_pct", "eps_beat"]].to_csv(filepath, index=False)
        logger.info("Saved %d EPS surprise records for %s", len(df), ticker)


# ---------------------------------------------------------------------------
# Shared: save call date metadata
# ---------------------------------------------------------------------------

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
    """Collect prices (yfinance), transcripts (Motley Fool), and EPS surprises (FMP).

    Transcripts are scraped from Motley Fool's public website since FMP transcript
    access requires a paid plan. Prices use yfinance (free). EPS surprises use
    FMP's free-tier endpoint (set FMP_API_KEY env var).

    Files already on disk are always skipped, so this is safe to re-run.

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

    logger.info("tickers=%d | years=%d-%d", len(tickers), min_year, max_year)

    logger.info("=== Fetching prices (yfinance) ===")
    fetch_fmp_prices(tickers, price_start, price_end, PRICES_DIR)

    logger.info("=== Fetching earnings press releases %d-%d (SEC EDGAR) ===", min_year, max_year)
    new_metadata = fetch_edgar_earnings_releases(tickers, min_year, max_year, TRANSCRIPTS_DIR)
    _save_fmp_dates(new_metadata)
    logger.info("Fetched %d new earnings release records", len(new_metadata))

    logger.info("=== Fetching EPS surprises (FMP free tier) ===")
    if FMP_API_KEY:
        fetch_all_eps_surprises(tickers, EPS_DIR)
    else:
        logger.warning("FMP_API_KEY not set — skipping EPS surprises (pipeline still works without them)")

    logger.info("Collection complete.")


if __name__ == "__main__":
    main_kaggle()

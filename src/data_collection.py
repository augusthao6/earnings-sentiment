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

_FOOL_BASE = "https://www.fool.com/earnings/call-transcripts"
_FOOL_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Motley Fool company slug for each supported ticker.
# Slug appears in the transcript URL:
#   https://www.fool.com/earnings/call-transcripts/YYYY/MM/DD/{SLUG}-qN-YYYY-earnings-call-transcript/
# Each ticker maps to a list of slug candidates tried in order.
# Motley Fool occasionally changes slug format (e.g. "apple-aapl" vs "apple-inc-aapl"),
# so we try the most common variants.
_FOOL_SLUGS: dict[str, list[str]] = {
    "AAPL": ["apple-aapl", "apple-inc-aapl"],
    "MSFT": ["microsoft-msft", "microsoft-corporation-msft"],
    "NVDA": ["nvidia-nvda", "nvidia-corporation-nvda"],
    "TSLA": ["tesla-tsla", "tesla-inc-tsla"],
    "META": ["meta-platforms-meta", "meta-platforms-inc-meta"],
    "AMZN": ["amazon-amzn", "amazon-com-amzn", "amazoncom-amzn"],
    "GOOGL": ["alphabet-googl", "alphabet-inc-googl"],
    "JPM": ["jpmorgan-chase-jpm", "jpmorgan-chase-co-jpm"],
    "BAC": ["bank-of-america-bac", "bank-of-america-corp-bac"],
    "GS": ["goldman-sachs-gs", "goldman-sachs-group-gs"],
    "XOM": ["exxon-mobil-xom", "exxonmobil-xom"],
    "CVX": ["chevron-cvx", "chevron-corporation-cvx"],
    "PFE": ["pfizer-pfe", "pfizer-inc-pfe"],
    "UNH": ["unitedhealth-group-unh", "unitedhealth-unh"],
    "AMD": ["advanced-micro-devices-amd"],
    "JNJ": ["johnson-johnson-jnj"],
}
# Keep a single-slug alias for backward compat
_FOOL_SLUG = {t: slugs[0] for t, slugs in _FOOL_SLUGS.items()}


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
# Motley Fool transcript scraper
# ---------------------------------------------------------------------------

def _parse_fool_transcript(soup: BeautifulSoup) -> str:
    """Extract transcript text from a Motley Fool article page."""
    # Try known article body container selectors (Motley Fool has changed layouts)
    candidates = [
        soup.find("div", class_=lambda c: c and "article-body" in c),
        soup.find("div", id="article-body"),
        soup.find("div", class_=lambda c: c and "tailwind-article-body" in c),
        soup.find("article"),
        soup.find("main"),
    ]
    for container in candidates:
        if container is None:
            continue
        paragraphs = container.find_all("p")
        text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        if len(text) > 2000:
            return text

    return ""


def _fool_try_url(url: str) -> str | None:
    """Fetch one Motley Fool URL and return transcript text, or None."""
    try:
        resp = requests.get(url, headers=_FOOL_HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        # Detect Cloudflare challenge page
        if "cf-browser-verification" in resp.text or "Checking your browser" in resp.text:
            logger.warning("Cloudflare challenge detected — Motley Fool is blocking automated requests")
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        text = _parse_fool_transcript(soup)
        return text if len(text) > 2000 else None
    except Exception:
        return None


def _ddg_find_fool_url(ticker: str, call_date: pd.Timestamp) -> tuple[str, int, int] | None:
    """Use DuckDuckGo HTML search to find the actual Motley Fool transcript URL.

    This is the fallback when direct URL construction fails (e.g. non-standard
    company slug, changed URL format). DuckDuckGo reliably indexes Fool transcripts.
    """
    import re
    from urllib.parse import quote, unquote, parse_qs, urlparse

    year = call_date.year
    query = f"site:fool.com earnings-call-transcript {ticker} {year}"
    ddg_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"

    try:
        time.sleep(2)
        resp = requests.get(ddg_url, headers=_FOOL_HEADERS, timeout=20)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")

        # DuckDuckGo HTML wraps destination URLs in a redirect link: href contains uddg=ENCODED_URL
        candidates = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "uddg=" not in href:
                continue
            try:
                actual = unquote(parse_qs(urlparse(href).query)["uddg"][0])
            except (KeyError, IndexError):
                continue
            if "fool.com" not in actual or "earnings-call-transcript" not in actual:
                continue
            candidates.append(actual)

        # Pick the result whose date in the URL is closest to call_date (within 3 days)
        best_url, best_delta = None, float("inf")
        for url in candidates:
            m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
            if not m:
                continue
            url_date = pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
            delta = abs((url_date - call_date).days)
            if delta <= 3 and delta < best_delta:
                best_delta = delta
                best_url = url

        if best_url:
            logger.info("DDG found transcript URL: %s", best_url)
            text = _fool_try_url(best_url)
            if text:
                m = re.search(r"-q(\d)-(\d{4})-earnings", best_url)
                if m:
                    return text, int(m.group(1)), int(m.group(2))
    except Exception as e:
        logger.debug("DuckDuckGo search failed for %s: %s", ticker, e)

    return None


def _find_fool_transcript(ticker: str, call_date: pd.Timestamp) -> tuple[str, int, int] | None:
    """Find and return a Motley Fool transcript for ticker near call_date.

    Step 1 — direct URL construction: fast, works for companies with known slugs.
    Step 2 — DuckDuckGo fallback: finds the real URL for companies whose slug
              doesn't match any of our candidates (e.g. MSFT URL format changed).

    Returns (text, fiscal_q, fiscal_year) or None.
    """
    slugs = _FOOL_SLUGS.get(ticker)

    # Step 1: direct URL construction (fast)
    if slugs:
        month = call_date.month
        if month in (1, 2, 3):
            q_order = [1, 2, 4, 3]
        elif month in (4, 5, 6):
            q_order = [2, 3, 1, 4]
        elif month in (7, 8, 9):
            q_order = [3, 4, 2, 1]
        else:
            q_order = [4, 1, 3, 2]
        years = [call_date.year, call_date.year + 1, call_date.year - 1]

        for day_offset in [0, 1]:
            d = call_date + pd.Timedelta(days=day_offset)
            for q in q_order:
                for y in years:
                    for slug in slugs:
                        url = (
                            f"{_FOOL_BASE}/{d.year}/{d.month:02d}/{d.day:02d}/"
                            f"{slug}-q{q}-{y}-earnings-call-transcript/"
                        )
                        time.sleep(0.25)
                        text = _fool_try_url(url)
                        if text:
                            logger.info("Found transcript (direct) at %s", url)
                            return text, q, y

    # Step 2: DuckDuckGo search fallback
    logger.info("Direct URL failed for %s ~%s — trying DuckDuckGo search", ticker, call_date.date())
    return _ddg_find_fool_url(ticker, call_date)


def fetch_motley_fool_transcripts(
    tickers: list[str],
    min_year: int,
    max_year: int,
    output_dir: Path,
) -> list[dict]:
    """Fetch earnings call transcripts from Motley Fool (free, public pages).

    Uses yfinance to discover earnings dates, then scrapes the public Motley Fool
    transcript page for each event. Transcripts are saved as {TICKER}_{YEAR}_Q{N}.txt
    — the same format the rest of the pipeline expects.

    Returns a list of metadata dicts (ticker, quarter, call_datetime, is_after_hours)
    compatible with _save_fmp_dates().
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_records = []

    for ticker in tickers:
        if ticker not in _FOOL_SLUG:
            logger.warning("No Motley Fool slug for %s — add it to _FOOL_SLUG to enable", ticker)
            continue

        # Get earnings dates from yfinance
        try:
            stock = yf.Ticker(ticker)
            earnings = stock.earnings_dates
            if earnings is None or earnings.empty:
                logger.warning("No earnings dates from yfinance for %s", ticker)
                continue
            earnings = earnings.reset_index()
            date_col = earnings.columns[0]
            # Normalize to timezone-naive
            if earnings[date_col].dt.tz is not None:
                earnings[date_col] = earnings[date_col].dt.tz_localize(None)
        except Exception as e:
            logger.warning("Could not get earnings dates for %s: %s", ticker, e)
            continue

        start = pd.Timestamp(f"{min_year}-01-01")
        end = min(pd.Timestamp(f"{max_year}-12-31"), pd.Timestamp.now())
        in_range = earnings[(earnings[date_col] >= start) & (earnings[date_col] <= end)]

        for _, row in in_range.iterrows():
            call_dt = row[date_col]
            # Calendar quarter from call month (used as fallback filename)
            cal_q = (call_dt.month - 1) // 3 + 1

            # Skip if we already have ANY transcript file for this ticker/period
            # (fiscal quarter may differ so check both)
            already_present = any(
                (output_dir / f"{ticker}_{call_dt.year}_Q{q}.txt").exists()
                for q in range(1, 5)
            )
            if already_present:
                logger.info("Skipping %s ~%s (transcript already on disk)", ticker, call_dt.date())
                continue

            logger.info("Fetching transcript: %s ~%s", ticker, call_dt.date())
            result = _find_fool_transcript(ticker, call_dt)

            if result is None:
                logger.warning("Transcript not found on Motley Fool: %s ~%s", ticker, call_dt.date())
                time.sleep(1)
                continue

            text, fiscal_q, fiscal_year = result
            filename = f"{ticker}_{fiscal_year}_Q{fiscal_q}.txt"
            (output_dir / filename).write_text(text, encoding="utf-8")
            logger.info("Saved %s (%d chars)", filename, len(text))

            # If yfinance gave us hour info use it; otherwise assume after-hours (most common)
            is_after_hours = call_dt.hour >= 16 if call_dt.hour != 0 else True
            metadata_records.append({
                "ticker": ticker,
                "quarter": f"{fiscal_year}-Q{fiscal_q}",
                "call_datetime": call_dt,
                "is_after_hours": is_after_hours,
            })

            time.sleep(2)  # polite crawl rate

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

    logger.info("=== Fetching transcripts %d-%d (Motley Fool) ===", min_year, max_year)
    new_metadata = fetch_motley_fool_transcripts(tickers, min_year, max_year, TRANSCRIPTS_DIR)
    _save_fmp_dates(new_metadata)
    logger.info("Fetched %d new transcript records", len(new_metadata))

    logger.info("=== Fetching EPS surprises (FMP free tier) ===")
    if FMP_API_KEY:
        fetch_all_eps_surprises(tickers, EPS_DIR)
    else:
        logger.warning("FMP_API_KEY not set — skipping EPS surprises (pipeline still works without them)")

    logger.info("Collection complete.")


if __name__ == "__main__":
    main_kaggle()

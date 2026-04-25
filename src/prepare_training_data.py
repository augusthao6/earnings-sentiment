"""Prepare data for fine-tuning FinBERT on earnings sentiment for options trading."""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import segment_transcript, tokenise_and_clean
from sentiment import load_lm_dictionary, score_lm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
PRICES_DIR = RAW_DIR / "prices"
TRANSCRIPTS_DIR = RAW_DIR / "transcripts"
EPS_DIR = RAW_DIR / "eps_surprises"
FMP_DATES_CSV = RAW_DIR / "fmp_transcript_dates.csv"
LM_DICT_PATH = RAW_DIR / "lm_dictionary.csv"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def discover_available_transcripts() -> pd.DataFrame:
    """Scan the transcripts directory and return a DataFrame of (ticker, quarter) pairs.

    This replaces the hardcoded TARGET_TICKERS list so the pipeline automatically
    picks up any new tickers added via main_fmp().
    """
    records = []
    for path in sorted(TRANSCRIPTS_DIR.glob("*.txt")):
        parts = path.stem.split("_")
        if len(parts) >= 3:
            records.append({"ticker": parts[0], "quarter": f"{parts[1]}-{parts[2]}"})
    df = pd.DataFrame(records).drop_duplicates()
    logger.info(
        "Found %d transcripts across %d tickers",
        len(df), df["ticker"].nunique(),
    )
    return df


def load_prices() -> dict[str, pd.DataFrame]:
    """Load all price CSVs found on disk (auto-detected, not hardcoded)."""
    prices = {}
    for path in sorted(PRICES_DIR.glob("*.csv")):
        if path.stem == ".gitkeep":
            continue
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            # Normalise column name — FMP saves 'Close', yfinance also 'Close'
            if "Close" not in df.columns and "adjClose" in df.columns:
                df = df.rename(columns={"adjClose": "Close"})
            prices[path.stem] = df
        except Exception as e:
            logger.warning("Could not load prices for %s: %s", path.stem, e)
    logger.info("Loaded price data for %d tickers", len(prices))
    return prices


def load_call_datetimes(tickers: list[str]) -> pd.DataFrame:
    """Load actual earnings call datetimes from the FMP transcript dates CSV.

    This file is written by main_fmp() during transcript download — each entry
    contains the exact datetime FMP recorded for that earnings call.
    """
    if not FMP_DATES_CSV.exists():
        raise RuntimeError(
            f"{FMP_DATES_CSV} not found.\n"
            "Run data collection first:\n"
            "    from src.data_collection import main_fmp\n"
            "    main_fmp(tickers=[...], min_year=2023, max_year=2026)"
        )
    df = pd.read_csv(FMP_DATES_CSV, parse_dates=["call_datetime"])
    df["call_date"] = df["call_datetime"].dt.normalize()
    # Filter to tickers that have transcripts on disk
    df = df[df["ticker"].isin(tickers)]
    logger.info("Loaded %d call date records for %d tickers", len(df), df["ticker"].nunique())
    return df[["ticker", "quarter", "call_datetime", "call_date", "is_after_hours"]]


def load_eps_surprises() -> dict[str, pd.DataFrame]:
    """Load EPS surprise CSVs for all tickers that have them."""
    eps = {}
    if not EPS_DIR.exists():
        return eps
    for path in sorted(EPS_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            eps[path.stem] = df.sort_values("date")
        except Exception as e:
            logger.warning("Could not load EPS surprises for %s: %s", path.stem, e)
    logger.info("Loaded EPS surprises for %d tickers", len(eps))
    return eps


def match_eps_surprise(
    eps_df: pd.DataFrame, call_date: pd.Timestamp, window_days: int = 7
) -> dict:
    """Find the EPS surprise record closest to call_date within window_days.

    EPS surprise controls for the headline beat/miss that drives most of the
    stock move. The residual sentiment variation is more meaningful once we
    know whether management beat or missed the number.
    """
    if eps_df is None or eps_df.empty:
        return {"eps_surprise_pct": 0.0, "eps_beat": 0}
    delta = (eps_df["date"] - call_date).abs()
    idx = delta.idxmin()
    if delta[idx].days > window_days:
        return {"eps_surprise_pct": 0.0, "eps_beat": 0}
    row = eps_df.loc[idx]
    surprise_pct = row.get("eps_surprise_pct", 0.0)
    return {
        "eps_surprise_pct": 0.0 if pd.isna(surprise_pct) else float(surprise_pct),
        "eps_beat": int(row.get("eps_beat", 0)),
    }


def load_sentiment_scores() -> pd.DataFrame:
    """Load pre-computed LM + FinBERT scores.

    If expanded transcripts are in TRANSCRIPTS_DIR but the old sentiment_scores.csv
    only covers the original 7 tickers, the merge in prepare_training_data will
    simply skip events without scores. Run notebook 02 (or add a sentiment script)
    to re-score new transcripts.
    """
    path = PROCESSED_DIR / "sentiment_scores.csv"
    if not path.exists():
        logger.warning(
            "sentiment_scores.csv not found — returning empty DataFrame. "
            "Run notebooks/02_sentiment_analysis.ipynb to generate it."
        )
        return pd.DataFrame(columns=["ticker", "quarter"])
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def estimate_implied_move(price_df: pd.DataFrame, day_0_date: pd.Timestamp) -> float:
    """Trailing 30-day realized vol × 1.25 as a proxy for options-implied move."""
    hist = price_df.loc[price_df.index <= day_0_date, "Close"].sort_index()
    if len(hist) < 31:
        return 0.05
    daily_returns = hist.pct_change().dropna()
    return float(daily_returns.iloc[-30:].std()) * 1.25


def calculate_returns(
    price_df: pd.DataFrame, call_date: pd.Timestamp, is_after_hours: bool
) -> dict:
    """Timing-corrected 1-day and 5-day returns after earnings.

    After-hours call: day 0 = call_date close (market hasn't reacted yet).
    Pre-market call:  day 0 = prior trading day close (market reacts same day).
    """
    price_idx = price_df.index.sort_values()
    candidates = (
        price_idx[price_idx <= call_date]
        if is_after_hours
        else price_idx[price_idx < call_date]
    )
    if len(candidates) == 0:
        return {}

    day_0_date = candidates[-1]
    day_0_close = price_df.loc[day_0_date, "Close"]
    future = price_idx[price_idx > day_0_date]
    if len(future) < 1:
        return {}

    return_1d = float((price_df.loc[future[0], "Close"] - day_0_close) / day_0_close)
    return_5d = (
        float((price_df.loc[future[4], "Close"] - day_0_close) / day_0_close)
        if len(future) >= 5
        else np.nan
    )
    return {
        "return_1d": return_1d,
        "return_5d": return_5d,
        "implied_move": estimate_implied_move(price_df, day_0_date),
        "day_0_date": day_0_date.strftime("%Y-%m-%d"),
    }


def score_qa_section(ticker: str, quarter: str, lm_dict: dict) -> dict:
    """Score only the Q&A section of a transcript with the LM dictionary.

    Q&A is less scripted than prepared remarks — management can't rehearse every
    analyst question, so tone here better reflects genuine confidence.
    """
    year, q = quarter.split("-")
    filename = f"{ticker}_{year}_{q}.txt"
    for search_dir in [PROCESSED_DIR, TRANSCRIPTS_DIR]:
        path = search_dir / filename
        if path.exists():
            text = path.read_text(encoding="utf-8")
            segments = segment_transcript(text)
            qa_text = segments.get("qa", segments.get("full", ""))
            tokens = tokenise_and_clean(qa_text)
            scores = score_lm(tokens, lm_dict)
            return {
                "qa_lm_positive": scores["positive"],
                "qa_lm_negative": scores["negative"],
                "qa_lm_net_sentiment": scores["net_sentiment"],
                "qa_lm_uncertainty": scores["uncertainty"],
            }
    logger.warning("Transcript not found for %s %s", ticker, quarter)
    return {k: 0.0 for k in ["qa_lm_positive", "qa_lm_negative", "qa_lm_net_sentiment", "qa_lm_uncertainty"]}


def add_qoq_features(df: pd.DataFrame) -> pd.DataFrame:
    """QoQ sentiment change — captures tone deterioration vs prior quarter.

    Raw scores are biased positive by management spin; the *change* is a cleaner
    signal. First quarter per ticker gets 0 (no prior period).
    """
    df = df.sort_values(["ticker", "quarter"]).copy()
    for col in [
        "lm_net_sentiment", "lm_uncertainty",
        "finbert_net_sentiment",
        "qa_lm_net_sentiment", "qa_lm_uncertainty",
    ]:
        df[f"{col}_qoq"] = df.groupby("ticker")[col].diff().fillna(0)
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare_training_data() -> pd.DataFrame:
    available = discover_available_transcripts()
    tickers = sorted(available["ticker"].unique().tolist())

    sentiment_df = load_sentiment_scores()
    prices = load_prices()
    call_dates = load_call_datetimes(tickers)  # FMP dates only
    eps_surprises = load_eps_surprises()
    lm_dict = load_lm_dictionary(LM_DICT_PATH)

    # Only keep events where we have both sentiment scores and a call date
    if sentiment_df.empty:
        # No pre-computed FinBERT scores yet — use only LM scores (computed here)
        base = available.merge(call_dates, on=["ticker", "quarter"], how="inner")
        # Add placeholder columns so the rest of the pipeline still works
        for col in ["lm_positive", "lm_negative", "lm_uncertainty",
                    "lm_net_sentiment", "finbert_positive", "finbert_negative", "finbert_net_sentiment"]:
            base[col] = 0.0
    else:
        base = sentiment_df.merge(call_dates, on=["ticker", "quarter"], how="inner")

    logger.info("Base dataset: %d events across %d tickers", len(base), base["ticker"].nunique())

    training_data = []
    for _, row in base.iterrows():
        ticker = row["ticker"]
        quarter = row["quarter"]
        if ticker not in prices:
            logger.warning("No price data for %s — skipping", ticker)
            continue

        returns = calculate_returns(prices[ticker], row["call_date"], row["is_after_hours"])
        if not returns:
            logger.warning("Could not calculate returns for %s %s — skipping", ticker, quarter)
            continue

        qa_scores = score_qa_section(ticker, quarter, lm_dict)
        eps = match_eps_surprise(
            eps_surprises.get(ticker), row["call_date"]
        )

        training_data.append({
            "ticker": ticker,
            "quarter": quarter,
            "call_datetime": row["call_datetime"],
            "is_after_hours": row["is_after_hours"],
            "day_0_date": returns["day_0_date"],
            # Full-transcript sentiment
            "lm_positive": row["lm_positive"],
            "lm_negative": row["lm_negative"],
            "lm_uncertainty": row["lm_uncertainty"],
            "lm_net_sentiment": row["lm_net_sentiment"],
            "finbert_positive": row["finbert_positive"],
            "finbert_negative": row["finbert_negative"],
            "finbert_net_sentiment": row["finbert_net_sentiment"],
            # Q&A-only scores — less scripted, more candid
            "qa_lm_positive": qa_scores["qa_lm_positive"],
            "qa_lm_negative": qa_scores["qa_lm_negative"],
            "qa_lm_net_sentiment": qa_scores["qa_lm_net_sentiment"],
            "qa_lm_uncertainty": qa_scores["qa_lm_uncertainty"],
            # EPS surprise — controls for headline beat/miss (Bias 3)
            "eps_surprise_pct": eps["eps_surprise_pct"],
            "eps_beat": eps["eps_beat"],
            # Returns
            "return_1d": returns["return_1d"],
            "return_5d": returns["return_5d"],
            "implied_move": returns["implied_move"],
            "realized_move": abs(returns["return_1d"]),
        })

    df = pd.DataFrame(training_data)
    df = add_qoq_features(df)

    # Label by per-ticker median: ~50/50 balance, asks "unusually large move for this stock?"
    ticker_medians = df.groupby("ticker")["realized_move"].transform("median")
    df["label"] = (df["realized_move"] > ticker_medians).astype(int)

    logger.info("Label distribution:\n%s", df["label"].value_counts().to_string())
    logger.info("EPS surprise coverage: %d/%d events have data",
                (df["eps_surprise_pct"] != 0).sum(), len(df))

    logger.info("Feature correlations with label:")
    key_features = [
        "qa_lm_net_sentiment", "qa_lm_uncertainty",
        "lm_net_sentiment_qoq", "finbert_net_sentiment_qoq",
        "eps_surprise_pct", "eps_beat",
    ]
    for feat in key_features:
        logger.info("  %-35s r = %.3f", feat, df[feat].corr(df["label"]))

    output_path = PROCESSED_DIR / "training_data.csv"
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)
    return df


if __name__ == "__main__":
    prepare_training_data()

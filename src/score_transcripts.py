"""Preprocess raw transcripts and score them with LM + FinBERT.

This script is the standalone equivalent of notebooks/01_preprocessing.ipynb
and notebooks/02_sentiment_analysis.ipynb. It is incremental — transcripts
already in data/processed/ or already in sentiment_scores.csv are skipped,
so it is safe to run after adding new transcripts via main_fmp().

Runtime: ~24 seconds per transcript for FinBERT on GPU (Colab T4).
         ~2-3 min per transcript on CPU. Use --lm-only to skip FinBERT.

Usage:
    python src/score_transcripts.py              # full LM + FinBERT
    python src/score_transcripts.py --lm-only    # LM dictionary only (fast)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import process_transcript
from sentiment import (
    load_lm_dictionary,
    score_lm,
    load_finbert,
    score_finbert_transcript,
)
from preprocessing import tokenise_and_clean

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRANSCRIPTS_DIR = RAW_DIR / "transcripts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LM_DICT_PATH = RAW_DIR / "lm_dictionary.csv"
SCORES_CSV = PROCESSED_DIR / "sentiment_scores.csv"


# ---------------------------------------------------------------------------
# Step 1: Preprocessing
# ---------------------------------------------------------------------------

def preprocess_all(force: bool = False) -> list[Path]:
    """Clean all raw transcripts and save to data/processed/.

    Returns the list of processed file paths (including already-processed ones).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(TRANSCRIPTS_DIR.glob("*.txt"))
    if not raw_files:
        logger.warning("No transcripts found in %s", TRANSCRIPTS_DIR)
        return []

    skipped, saved = 0, 0
    for path in raw_files:
        out_path = PROCESSED_DIR / path.name
        if out_path.exists() and not force:
            skipped += 1
            continue
        result = process_transcript(path)
        out_path.write_text(result["cleaned_text"], encoding="utf-8")
        saved += 1

    logger.info(
        "Preprocessing: %d saved, %d already existed (use --force to reprocess all)",
        saved, skipped,
    )
    return sorted(PROCESSED_DIR.glob("*.txt"))


# ---------------------------------------------------------------------------
# Step 2: Sentiment scoring
# ---------------------------------------------------------------------------

def load_existing_scores() -> pd.DataFrame:
    if SCORES_CSV.exists():
        df = pd.read_csv(SCORES_CSV)
        logger.info("Loaded %d existing scores from %s", len(df), SCORES_CSV)
        return df
    return pd.DataFrame()


def score_all(processed_files: list[Path], lm_only: bool = False) -> pd.DataFrame:
    """Score all processed transcripts not yet in sentiment_scores.csv.

    Returns the complete (old + new) scores DataFrame.
    """
    existing = load_existing_scores()
    already_scored = set()
    if not existing.empty:
        already_scored = set(
            zip(existing["ticker"].tolist(), existing["quarter"].tolist())
        )

    # Identify which files need scoring
    to_score = []
    for path in processed_files:
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        ticker, year, q = parts[0], parts[1], parts[2]
        quarter = f"{year}-{q}"
        if (ticker, quarter) not in already_scored:
            to_score.append((path, ticker, quarter))

    if not to_score:
        logger.info("All %d transcripts already scored — nothing to do.", len(processed_files))
        return existing

    logger.info("Scoring %d new transcripts (lm_only=%s)", len(to_score), lm_only)

    lm_dict = load_lm_dictionary(LM_DICT_PATH)
    tokenizer, finbert_model = (None, None) if lm_only else load_finbert()

    new_rows = []
    start = time.time()
    for i, (path, ticker, quarter) in enumerate(to_score):
        text = path.read_text(encoding="utf-8")
        tokens = tokenise_and_clean(text)
        lm_scores = score_lm(tokens, lm_dict)

        row = {
            "ticker": ticker,
            "quarter": quarter,
            "lm_positive": lm_scores["positive"],
            "lm_negative": lm_scores["negative"],
            "lm_uncertainty": lm_scores["uncertainty"],
            "lm_litigious": lm_scores["litigious"],
            "lm_constraining": lm_scores["constraining"],
            "lm_net_sentiment": lm_scores["net_sentiment"],
            "token_count": len(tokens),
        }

        if not lm_only:
            fb = score_finbert_transcript(text, tokenizer, finbert_model)
            row.update({
                "finbert_positive": fb["positive"],
                "finbert_negative": fb["negative"],
                "finbert_neutral": fb["neutral"],
                "finbert_net_sentiment": fb["net_sentiment"],
            })
        else:
            row.update({
                "finbert_positive": 0.0,
                "finbert_negative": 0.0,
                "finbert_neutral": 0.0,
                "finbert_net_sentiment": 0.0,
            })

        new_rows.append(row)

        elapsed = time.time() - start
        rate = elapsed / (i + 1)
        remaining = rate * (len(to_score) - i - 1)
        logger.info(
            "[%d/%d] %s %s  (%.1fs elapsed, ~%.0fs remaining)",
            i + 1, len(to_score), ticker, quarter, elapsed, remaining,
        )

        # Save a checkpoint every 10 transcripts in case the session times out
        if (i + 1) % 10 == 0:
            _save_scores(existing, new_rows)
            logger.info("Checkpoint saved (%d new rows so far)", len(new_rows))

    return _save_scores(existing, new_rows)


def _save_scores(existing: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    if not new_rows:
        return existing
    new_df = pd.DataFrame(new_rows)
    combined = (
        pd.concat([existing, new_df], ignore_index=True)
        if not existing.empty
        else new_df
    )
    combined = combined.drop_duplicates(subset=["ticker", "quarter"], keep="last")
    combined = combined.sort_values(["ticker", "quarter"]).reset_index(drop=True)
    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(SCORES_CSV, index=False)
    logger.info("Saved %d total score rows to %s", len(combined), SCORES_CSV)
    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess transcripts and score sentiment.")
    parser.add_argument(
        "--lm-only", action="store_true",
        help="Skip FinBERT (LM dictionary only). Fast but produces 0.0 for finbert columns.",
    )
    parser.add_argument(
        "--force-preprocess", action="store_true",
        help="Re-preprocess all transcripts even if already in data/processed/.",
    )
    args = parser.parse_args()

    if args.lm_only:
        logger.info("Running in LM-only mode (FinBERT skipped)")
    else:
        logger.info(
            "Running full LM + FinBERT mode. "
            "This takes ~24s per transcript on GPU. "
            "Pass --lm-only to skip FinBERT."
        )

    processed = preprocess_all(force=args.force_preprocess)
    if not processed:
        logger.error("No processed transcripts found — nothing to score.")
        return

    score_all(processed, lm_only=args.lm_only)
    logger.info("Done. Run prepare_training_data.py next.")


if __name__ == "__main__":
    main()

"""Build fusion model combining transcript sentiment and structured features."""

import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def load_training_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "training_data.csv"
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and labels for the fusion model.

    Feature groups:
    - Full-transcript sentiment (LM + FinBERT) — broad tone signal
    - Q&A-only LM sentiment — less scripted, more candid under analyst pressure
    - QoQ sentiment change — captures tone deterioration/improvement vs prior quarter,
      which is more informative than the raw level (management always sounds positive)
    - Market context (implied vol, sector, market cap proxy)
    """
    features = df[[
        # Full-transcript sentiment
        "lm_positive", "lm_negative", "lm_uncertainty", "lm_net_sentiment",
        "finbert_positive", "finbert_negative", "finbert_net_sentiment",
        # Q&A-only sentiment (notebook found prepared remarks are heavily scripted)
        "qa_lm_positive", "qa_lm_negative", "qa_lm_net_sentiment", "qa_lm_uncertainty",
        # QoQ change: did tone improve or worsen vs last quarter?
        "lm_net_sentiment_qoq", "lm_uncertainty_qoq",
        "finbert_net_sentiment_qoq",
        "qa_lm_net_sentiment_qoq", "qa_lm_uncertainty_qoq",
        # EPS surprise — controls for headline beat/miss (Bias 3)
        # Fills 0 when not available so the model degrades gracefully
        "eps_surprise_pct", "eps_beat",
        # Market context
        "implied_move",
    ]].copy()

    features["market_cap_proxy"] = df["ticker"].map({
        "AAPL": 3e12, "MSFT": 3e12, "AMZN": 1.5e12, "GOOGL": 1.8e12,
        "JPM": 5e11, "JNJ": 4e11, "BAC": 3e11,
    })
    features["sector"] = df["ticker"].map({
        "AAPL": "tech", "MSFT": "tech", "AMZN": "tech", "GOOGL": "tech",
        "JPM": "finance", "JNJ": "health", "BAC": "finance",
    })
    features = pd.get_dummies(features, columns=["sector"])

    labels = df["label"]
    return features, labels


def train_fusion_model():
    df = load_training_data()
    features, labels = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Fusion model accuracy: %.4f", acc)
    print(classification_report(y_test, y_pred))

    # Log top feature importances so you can see what's actually driving predictions
    importances = pd.Series(model.feature_importances_, index=features.columns)
    top10 = importances.nlargest(10)
    logger.info("Top 10 feature importances:\n%s", top10.to_string())

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "fusion_model.pkl")
    logger.info("Fusion model saved to models/fusion_model.pkl")


if __name__ == "__main__":
    train_fusion_model()

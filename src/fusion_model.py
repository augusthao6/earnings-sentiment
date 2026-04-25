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
    """Load training data with features."""
    path = PROCESSED_DIR / "training_data.csv"
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and labels for fusion model."""
    # Features: sentiment scores + structured
    features = df[[
        'lm_positive', 'lm_negative', 'lm_net_sentiment',
        'finbert_positive', 'finbert_negative', 'finbert_net_sentiment',
        'implied_move'  # proxy
    ]].copy()
    # Add some dummy structured features
    features['market_cap_proxy'] = df['ticker'].map({
        'AAPL': 3e12, 'MSFT': 3e12, 'AMZN': 1.5e12, 'GOOGL': 1.8e12,
        'JPM': 5e11, 'JNJ': 4e11, 'BAC': 3e11
    })
    features['sector'] = df['ticker'].map({
        'AAPL': 'tech', 'MSFT': 'tech', 'AMZN': 'tech', 'GOOGL': 'tech',
        'JPM': 'finance', 'JNJ': 'health', 'BAC': 'finance'
    })
    features = pd.get_dummies(features, columns=['sector'])
    labels = df['label']
    return features, labels


def train_fusion_model():
    """Train the fusion model."""
    df = load_training_data()
    features, labels = prepare_features(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Fusion model accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "fusion_model.pkl")
    logger.info("Fusion model saved to models/fusion_model.pkl")


if __name__ == "__main__":
    train_fusion_model()
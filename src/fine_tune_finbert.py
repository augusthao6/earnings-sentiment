"""Fine-tune FinBERT on earnings call Q&A sections to predict straddle signal."""

import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import segment_transcript

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
TRANSCRIPTS_DIR = RAW_DIR / "transcripts"
MODELS_DIR = PROJECT_ROOT / "models"

MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 2e-5


class EarningsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_qa_text(ticker: str, quarter: str) -> str:
    """Load only the Q&A section of an earnings call transcript.

    Using Q&A instead of the full transcript has two advantages:
    1. The first 512 tokens of a full transcript is just the IR intro — useless signal.
       The first 512 tokens of the Q&A is actual analyst questions and management responses.
    2. Q&A is less scripted: management can't rehearse every analyst question, so
       genuine confidence and hedging language surfaces more clearly here.
    """
    year, q = quarter.split("-")
    filename = f"{ticker}_{year}_{q}.txt"
    for search_dir in [PROCESSED_DIR, TRANSCRIPTS_DIR]:
        path = search_dir / filename
        if path.exists():
            text = path.read_text(encoding="utf-8")
            segments = segment_transcript(text)
            return segments.get("qa", segments.get("full", text))
    return ""


def load_training_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "training_data.csv"
    return pd.read_csv(path)


def prepare_data_for_finetuning():
    df = load_training_data()
    texts, labels = [], []
    missing = 0
    for _, row in df.iterrows():
        text = load_qa_text(row["ticker"], row["quarter"])
        if text:
            texts.append(text)
            labels.append(row["label"])
        else:
            missing += 1
    if missing:
        logger.warning("%d transcripts not found and skipped", missing)
    logger.info("Loaded %d Q&A sections for fine-tuning", len(texts))
    return texts, labels


def fine_tune_finbert():
    texts, labels = prepare_data_for_finetuning()
    if not texts:
        logger.error("No training data found")
        return

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    logger.info(
        "Train: %d samples (label 1: %d), Val: %d samples (label 1: %d)",
        len(train_labels), sum(train_labels),
        len(val_labels), sum(val_labels),
    )

    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    )

    train_dataset = EarningsDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = EarningsDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)
    model.to(device)

    # Class weights: up-weight label 1 proportionally to its scarcity
    n_class0 = train_labels.count(0)
    n_class1 = train_labels.count(1)
    class_weight = torch.tensor(
        [1.0, n_class0 / max(n_class1, 1)], dtype=torch.float
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight)
    logger.info("Class weights: [0]=1.0, [1]=%.2f", n_class0 / max(n_class1, 1))

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(batch_labels.cpu().numpy())

        val_acc = accuracy_score(val_true, val_preds)
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_acc=%.4f",
            epoch + 1, EPOCHS, train_loss / len(train_loader), val_acc,
        )

    MODELS_DIR.mkdir(exist_ok=True)
    model.save_pretrained(MODELS_DIR / "finbert_finetuned")
    tokenizer.save_pretrained(MODELS_DIR / "finbert_finetuned")
    logger.info("Model saved to models/finbert_finetuned")

    print(classification_report(val_true, val_preds, zero_division=0))


if __name__ == "__main__":
    fine_tune_finbert()

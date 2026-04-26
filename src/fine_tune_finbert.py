"""Fine-tune FinBERT on earnings press releases to predict 3-class straddle signal.

Labels (from prepare_training_data.py tertile split):
  0 = sell straddle  (small realized move — vol was overpriced)
  1 = do nothing     (middling move — ambiguous)
  2 = buy straddle   (large realized move — vol was underpriced)

FinBERT's pretrained head is already 3-class (positive / negative / neutral),
so num_labels=3 loads with no size mismatch and transfers the BERT backbone
weights directly. The classifier head is re-initialized during fine-tuning.
"""

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
EPOCHS = 5
LEARNING_RATE = 2e-5
NUM_LABELS = 3  # 0=sell, 1=hold, 2=buy
PATIENCE = 2    # early stopping: halt if val_acc doesn't improve for this many epochs


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
            labels.append(int(row["label"]))
        else:
            missing += 1
    if missing:
        logger.warning("%d transcripts not found and skipped", missing)
    logger.info("Loaded %d sections for fine-tuning", len(texts))
    return texts, labels


def fine_tune_finbert():
    texts, labels = prepare_data_for_finetuning()
    if not texts:
        logger.error("No training data found")
        return

    label_counts = {c: labels.count(c) for c in sorted(set(labels))}
    logger.info("Label distribution: %s", label_counts)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    logger.info(
        "Train: %d samples  Val: %d samples",
        len(train_labels), len(val_labels),
    )

    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # num_labels=3 matches FinBERT's pretrained head exactly — no size mismatch
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS
    )

    train_dataset = EarningsDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = EarningsDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)
    model.to(device)

    # Per-class inverse-frequency weights to handle class imbalance
    n = len(train_labels)
    class_weight = torch.tensor(
        [n / (NUM_LABELS * max(train_labels.count(c), 1)) for c in range(NUM_LABELS)],
        dtype=torch.float,
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight)
    logger.info(
        "Class weights: [0=sell]=%.2f [1=hold]=%.2f [2=buy]=%.2f",
        *class_weight.tolist(),
    )

    best_val_acc = -1.0
    best_epoch = 0
    epochs_no_improve = 0

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
            "Epoch %d/%d  train_loss=%.4f  val_acc=%.4f%s",
            epoch + 1, EPOCHS,
            train_loss / len(train_loader),
            val_acc,
            "  [best]" if val_acc > best_val_acc else "",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            MODELS_DIR.mkdir(exist_ok=True)
            model.save_pretrained(MODELS_DIR / "finbert_finetuned")
            tokenizer.save_pretrained(MODELS_DIR / "finbert_finetuned")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                logger.info("Early stopping at epoch %d (best was epoch %d)", epoch + 1, best_epoch)
                break

    logger.info("Best val_acc=%.4f at epoch %d — model saved to models/finbert_finetuned", best_val_acc, best_epoch)
    print(classification_report(val_true, val_preds,
                                target_names=["sell(0)", "hold(1)", "buy(2)"],
                                zero_division=0))


if __name__ == "__main__":
    fine_tune_finbert()

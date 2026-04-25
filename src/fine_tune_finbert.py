"""Fine-tune FinBERT on earnings transcripts to predict straddle signal."""

import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

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
EPOCHS = 3
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
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_transcript_text(ticker: str, quarter: str) -> str:
    """Load transcript text for a ticker and quarter."""
    filename = f"{ticker}_{quarter.replace('-', '_')}.txt"
    path = TRANSCRIPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding='utf-8')
    else:
        # Check processed
        processed_path = PROCESSED_DIR / filename
        if processed_path.exists():
            return processed_path.read_text(encoding='utf-8')
    return ""


def load_training_data() -> pd.DataFrame:
    """Load the prepared training data."""
    path = PROCESSED_DIR / "training_data.csv"
    return pd.read_csv(path)


def prepare_data_for_finetuning():
    """Prepare texts and labels for fine-tuning."""
    df = load_training_data()
    texts = []
    labels = []
    for _, row in df.iterrows():
        text = load_transcript_text(row['ticker'], row['quarter'])
        if text:
            texts.append(text)
            labels.append(row['label'])
    return texts, labels


def fine_tune_finbert():
    """Fine-tune FinBERT on the earnings data."""
    texts, labels = prepare_data_for_finetuning()
    if not texts:
        logger.error("No training data found")
        return

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Load tokenizer and model
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Create datasets
    train_dataset = EarningsDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = EarningsDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_true, val_preds)
        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

    # Save model
    MODELS_DIR.mkdir(exist_ok=True)
    model.save_pretrained(MODELS_DIR / "finbert_finetuned")
    tokenizer.save_pretrained(MODELS_DIR / "finbert_finetuned")
    logger.info("Model saved to models/finbert_finetuned")

    # Final evaluation
    print(classification_report(val_true, val_preds))


if __name__ == "__main__":
    fine_tune_finbert()
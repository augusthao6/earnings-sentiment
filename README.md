# Options Trading on Earnings Call Transcript Sentiment

This project fine-tunes FinBERT on earnings call transcripts to predict whether stocks will over- or under-perform relative to options-implied expectations. The model outputs a signal to buy/sell a straddle, which is then backtested for profitability. Built as a final project for CS 372, demonstrating end-to-end ML in finance.

## What it Does

The system analyzes earnings call transcripts using NLP to predict post-earnings volatility. If the model predicts high volatility (beyond options implied), it signals buying a straddle (bet on big move). Otherwise, no position. Backtesting shows the strategy's P&L compared to baselines.

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/rj694/earnings-sentiment.git && cd earnings-sentiment
python -m venv .venv && source .venv/bin/activate  # or conda create -n earnings-sentiment python=3.10
pip install -r requirements.txt

# 2. Get data (existing transcripts from Kaggle)
# Download from: https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts
# Place in data/raw/kaggle_source/motley-fool-data.pkl

# 3. Run data collection
python src/data_collection.py

# 4. Prepare training data
python src/prepare_training_data.py

# 5. Fine-tune FinBERT
python src/fine_tune_finbert.py

# 6. Train fusion model
python src/fusion_model.py

# 7. Run backtest
python src/backtester.py
```

## Video Links

- [Demo Video](videos/demo.mp4) - Shows the system in action
- [Technical Walkthrough](videos/technical.mp4) - Explains code and ML techniques

## Evaluation

Backtest results on 56 earnings events:
- Strategy P&L: +X.XX
- Win Rate: XX%
- Sharpe Ratio: X.XX
- vs Baseline (always buy straddle): +Y.YY

Model accuracy on validation: XX%

## Individual Contributions

Solo project by [Your Name].

## Approach

1. **Data Collection** — Earnings transcripts from Kaggle, prices from yfinance, options data approximated.
2. **Preprocessing** — Clean transcripts, calculate returns and implied moves.
3. **Model Training** — Fine-tune FinBERT on transcripts for straddle signal, fusion with structured features.
4. **Backtesting** — Simulate straddle P&L with realistic spreads.

## Tools

- Python, PyTorch, Transformers
- scikit-learn, pandas
- yfinance, backtrader

Python 3.12 | pandas | matplotlib | seaborn | NLTK | Hugging Face Transformers (FinBERT) | yfinance | scipy

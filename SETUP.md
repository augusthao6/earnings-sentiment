# Setup Instructions

## Prerequisites

- Python 3.8+
- Git

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rj694/earnings-sentiment.git
   cd earnings-sentiment
   ```

2. Create virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Setup

1. Download earnings transcripts from Kaggle:
   - Go to: https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts
   - Download the pickle file
   - Place it in `data/raw/kaggle_source/motley-fool-data.pkl`

2. The script will automatically download stock prices via yfinance.

## Running the Project

Follow the Quick Start in README.md.

## API Keys

No API keys required for basic functionality. For extended data collection (social media), you may need:
- StockTwits API key
- Reddit API credentials
- FinancialModelingPrep API key (free tier available)

## Troubleshooting

- If yfinance fails, try updating: `pip install --upgrade yfinance`
- For GPU training, ensure PyTorch with CUDA is installed
- If FinBERT download fails, check internet connection
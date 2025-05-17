# 📰📈 News Sentiment Market Model

A multifactor linear regression model that explores the relationship between daily financial news sentiment and short-term S&P 500 returns. Built for the IEOR 198: Introduction to Quantitative Finance final project at UC Berkeley.

## 📊 Project Overview

This project tests whether aggregated daily sentiment from news headlines can predict market movements. It uses:

- 📌 **Daily News Sentiment** (via VADER)
- 🔁 **Momentum** (previous day's return)
- 📦 **Volume** (standardized)

Regression is performed on S&P 500 ETF (SPY) return data between 2020–2024.

## 📁 Files

| File | Description |
|------|-------------|
| `main.py` | Main script that loads data, computes sentiment, and runs regression |
| `cnbc_headlines.csv`, `guardian_headlines.csv`, `reuters_headlines.csv` | Datasets containing historical news headlines |
| `merged_sentiment_spy_data.csv` | Final cleaned dataset with sentiment + return features |
| `sentiment_vs_return.png` | Scatter plot of sentiment vs return |
| `Sentiment_Factor_Model_Report.tex` | Full LaTeX report with abstract, code, and results |
| `requirements.txt` | Python dependencies for easy setup |

## 🚀 How to Reproduce

1. **Clone this repo**
   ```bash
   git clone https://github.com/YOURUSERNAME/news-sentiment-market-model
   cd news-sentiment-market-model

# Final Project: Multifactor Model Using Historical News Sentiment and Momentum

import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Helper function to load and clean each CSV file
def load_and_clean(path):
    d = pd.read_csv(path)
    d.columns = d.columns.str.lower()

    # Normalize column names
    if 'headlines' in d.columns:
        d.rename(columns={'headlines': 'headline'}, inplace=True)
    if 'text' in d.columns:
        d.rename(columns={'text': 'headline'}, inplace=True)
    if 'published' in d.columns:
        d.rename(columns={'published': 'date'}, inplace=True)
    if 'time' in d.columns:
        d.rename(columns={'time': 'date'}, inplace=True)

    # Convert to datetime and drop bad rows
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d = d.dropna(subset=['date', 'headline'])
    return d

# Load all datasets
df1 = load_and_clean("cnbc_headlines.csv")
df2 = load_and_clean("guardian_headlines.csv")
df3 = load_and_clean("reuters_headlines.csv")

# Combine them
df = pd.concat([df1, df2, df3], ignore_index=True)

# Step 2: Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
df['sentiment'] = df['headline'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Step 3: Aggregate sentiment by day
daily_sentiment = df.groupby(df['date'].dt.date)['sentiment'].mean()
daily_sentiment = daily_sentiment.to_frame().rename(columns={'sentiment': 'daily_sentiment'})
daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
daily_sentiment.index.name = 'Date'

# Step 4: Download SPY historical price data
spy_data = yf.download("SPY", start=daily_sentiment.index.min(), end=daily_sentiment.index.max())
print("SPY columns:", spy_data.columns)

# ✅ Flatten MultiIndex columns if they exist
if isinstance(spy_data.columns, pd.MultiIndex):
    spy_data.columns = [f"{col[0]}_{col[1]}" for col in spy_data.columns]

# ✅ Reset index so we can merge on 'Date' as a column
spy_data = spy_data.reset_index()
spy_data['Date'] = pd.to_datetime(spy_data['Date'])

# Calculate daily returns using the correct flattened column name
spy_data['daily_return'] = spy_data['Close_SPY'].pct_change()

# Step 5: Merge sentiment with SPY returns
sentiment_df = daily_sentiment.reset_index()
merged = pd.merge(spy_data, sentiment_df, on='Date', how='inner')

# Step 6: Add additional factor(s)
# Lagged return (momentum)
merged['prev_return'] = merged['daily_return'].shift(1)

# Normalize volume
merged['volume_scaled'] = (merged['Volume_SPY'] - merged['Volume_SPY'].mean()) / merged['Volume_SPY'].std()

# Drop rows with missing values
merged.dropna(inplace=True)

# Step 7: Run regression with multiple factors
X = merged[['daily_sentiment', 'prev_return', 'volume_scaled']]
X = sm.add_constant(X)
y = merged['daily_return']
model = sm.OLS(y, X).fit()
print(model.summary())

# Step 8: Visualize sentiment vs return
plt.figure(figsize=(10, 5))
plt.scatter(merged['daily_sentiment'], merged['daily_return'], alpha=0.5)
plt.xlabel("Daily Sentiment")
plt.ylabel("SPY Daily Return")
plt.title("News Sentiment vs SPY Daily Return")
plt.grid(True)
plt.tight_layout()
plt.savefig("sentiment_vs_return.png")
plt.show()

# Save merged dataset
merged.to_csv("merged_sentiment_spy_data.csv", index=False)
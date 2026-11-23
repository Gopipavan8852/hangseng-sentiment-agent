
# Hang Seng Social Sentiment & Price Prediction Agent

An AI-powered stock analysis system that combines social media sentiment, financial news, and historical market data to analyze stocks from the Hong Kong and South Korean markets.

## Project Overview

This project collects financial discussions and news, performs multilingual sentiment analysis, processes historical stock prices, and generates experimental next-trading-day return predictions.

## Markets Covered

### Hong Kong
- HSBC Holdings (0005.HK)
- Tencent (0700.HK)
- Alibaba (9988.HK)
- Xiaomi (1810.HK)
- AIA Group (1299.HK)

### South Korea
- Samsung Electronics (005930.KS)
- SK Hynix (000660.KS)
- NAVER (035420.KS)
- Kakao (035720.KS)
- Hyundai Motor (005380.KS)

## Key Features

- Financial news collection using Exa.
- Social media data collection using Apify.
- Multilingual sentiment classification using Transformers.
- Historical stock market data using Yahoo Finance.
- Market feature engineering.
- Next-trading-day return prediction.
- Model comparison and evaluation.
- Interactive Streamlit dashboard.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Transformers
- PyTorch
- Streamlit
- Apify
- Exa
- Yahoo Finance
- Matplotlib

## Project Architecture

1. Data Collection
2. Data Processing
3. Sentiment Analysis
4. Feature Engineering
5. Model Training and Evaluation
6. Next-Day Prediction
7. Streamlit Dashboard

## Model Evaluation

Models evaluated:
- Linear Regression
- Random Forest Regressor
- Zero Return Baseline

The full historical evaluation showed that the zero-return baseline outperformed the tested machine-learning models on MAE and RMSE.

A separate recent-period experiment used only 13 records with available sentiment, split into 6 training and 7 testing records. Its results are exploratory and not statistically reliable.

Sentiment features did not demonstrate a measurable improvement in these experiments.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the data collection and processing scripts as configured in the project.

Start the dashboard:

```bash
python -m streamlit run app.py
```

## Output Files

Important generated files include:

- `data/market_data.csv`
- `data/news.csv`
- `data/social_data.csv`
- `data/sentiment_results.csv`
- `data/prediction_dataset.csv`
- `reports/model_metrics.csv`
- `reports/model_comparison.csv`
- `reports/recent_sentiment_evaluation.csv`
- `reports/next_day_predictions.csv`

## Limitations

- Social sentiment coverage is limited to a short collection period.
- Some social sources returned few usable ticker-matched records.
- Sentiment data is sparse compared with historical market data.
- Current prediction models did not outperform the zero-return baseline in the historical evaluation.
- Results are experimental and are not investment advice.

## Author

A.GopipavanS

## Disclaimer

This project is developed for educational and research purposes only. Predictions are experimental and should not be used as financial advice.

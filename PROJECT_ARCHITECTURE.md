
# Project Architecture
## Hang Seng Social Sentiment & Price Prediction Agent

```mermaid
flowchart TD
    A[Data Collection]

    A --> B[Social Media Data]
    A --> C[Financial News]
    A --> D[Historical Stock Prices]

    B --> E[Apify]
    C --> F[Exa API]
    D --> G[Yahoo Finance]

    E --> H[Data Processing]
    F --> H
    G --> H

    H --> I[Data Cleaning and Ticker Matching]

    I --> J[Sentiment Analysis]
    J --> K[Multilingual Transformer Model]

    I --> L[Market Feature Engineering]
    K --> M[Sentiment Feature Engineering]

    L --> N[Prediction Dataset]
    M --> N

    N --> O[Model Training and Evaluation]

    O --> P[Linear Regression]
    O --> Q[Random Forest]
    O --> R[Zero Return Baseline]

    P --> S[Next Trading Day Predictions]
    Q --> S
    R --> T[Model Comparison]
    S --> T

    T --> U[Streamlit Dashboard]
    U --> V[Visual Reports and Analysis]
```

## Architecture Explanation

### 1. Data Collection
- Apify collects social media discussions.
- Exa collects financial news.
- Yahoo Finance provides historical OHLCV market data.

### 2. Data Processing
- Cleans collected records.
- Identifies stock ticker mentions.
- Combines news, social and market datasets.

### 3. Sentiment Analysis
- Uses a multilingual Transformer model.
- Classifies financial text into sentiment categories.
- Calculates sentiment scores and confidence.

### 4. Feature Engineering
- Calculates historical returns.
- Generates moving averages and volatility features.
- Creates lagged sentiment features.

### 5. Model Training
- Trains Linear Regression and Random Forest models.
- Compares predictions with a zero-return baseline.
- Evaluates MAE, RMSE and directional accuracy.

### 6. Dashboard
- Displays market information.
- Shows sentiment analysis.
- Presents experimental stock predictions.
- Displays model evaluation results.

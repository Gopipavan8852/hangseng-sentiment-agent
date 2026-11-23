
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "prediction_dataset.csv"
OUTPUT_PATH = BASE_DIR / "reports" / "recent_sentiment_evaluation.csv"

MARKET_FEATURES = [
    "open", "high", "low", "close", "volume",
    "daily_return", "momentum_5", "sma_ratio",
    "volatility_5", "volume_change", "daily_range"
]

SENTIMENT_FEATURES = [
    "sentiment_score_lag1",
    "sentiment_count_lag1",
    "avg_confidence_lag1"
]

print("=" * 65)
print("RECENT SENTIMENT MODEL EVALUATION")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(
    df["date"], utc=True, errors="coerce"
)

df["target_return"] = pd.to_numeric(
    df["target_return"], errors="coerce"
)

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=["date", "ticker", "target_return"]
).copy()

# Keep only records with actual sentiment observations.
# A zero sentiment score is valid if sentiment_count is positive.
df = df[df["sentiment_count_lag1"] > 0].copy()

market_features = [
    col for col in MARKET_FEATURES if col in df.columns
]

sentiment_features = [
    col for col in SENTIMENT_FEATURES if col in df.columns
]

df = df.sort_values("date").reset_index(drop=True)

print("\nRows with sentiment:", len(df))
print("Tickers:", df["ticker"].nunique())
print("Dates:", df["date"].nunique())

if len(df) < 10:
    raise ValueError(
        "Too few sentiment records for this experiment."
    )

# Chronological split, not random split.
dates = sorted(df["date"].unique())

split_index = int(len(dates) * 0.70)

if split_index <= 0 or split_index >= len(dates):
    raise ValueError(
        "Not enough unique dates for train/test evaluation."
    )

split_date = dates[split_index]

train = df[df["date"] < split_date].copy()
test = df[df["date"] >= split_date].copy()

print("\nTraining rows:", len(train))
print("Testing rows:", len(test))
print("Test begins:", split_date)

if train.empty or test.empty:
    raise ValueError(
        "Training or testing split is empty."
    )

# Use the same test rows for every model.
X_train_market = train[market_features + ["ticker"]]
X_test_market = test[market_features + ["ticker"]]

X_train_combined = train[
    market_features + sentiment_features + ["ticker"]
]

X_test_combined = test[
    market_features + sentiment_features + ["ticker"]
]

y_train = train["target_return"].to_numpy() * 100
y_test = test["target_return"].to_numpy() * 100


def create_pipeline(model_type, numeric_features):

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    category_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("category", category_pipeline, ["ticker"])
    ])

    if model_type == "Linear Regression":
        estimator = LinearRegression()
    else:
        estimator = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

    return Pipeline([
        ("preprocess", preprocess),
        ("model", estimator)
    ])


results = []

experiments = [
    (
        "Linear Regression - Market Only",
        "Linear Regression",
        market_features,
        X_train_market,
        X_test_market
    ),
    (
        "Linear Regression - Market + Sentiment",
        "Linear Regression",
        market_features + sentiment_features,
        X_train_combined,
        X_test_combined
    ),
    (
        "Random Forest - Market Only",
        "Random Forest",
        market_features,
        X_train_market,
        X_test_market
    ),
    (
        "Random Forest - Market + Sentiment",
        "Random Forest",
        market_features + sentiment_features,
        X_train_combined,
        X_test_combined
    )
]

for name, model_type, features, X_train, X_test in experiments:

    if len(train) < 5:
        print("Skipping due to insufficient training data:", name)
        continue

    model = create_pipeline(model_type, features)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    directional_accuracy = (
        np.mean(np.sign(y_test) == np.sign(predictions)) * 100
    )

    results.append({
        "model": name,
        "train_records": len(train),
        "test_records": len(test),
        "MAE_Return_Percent": mae,
        "RMSE_Return_Percent": rmse,
        "Directional_Accuracy_Percent": directional_accuracy
    })

    print("\n", name)
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("Directional accuracy:", round(directional_accuracy, 2))


results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 65)
print("RECENT SENTIMENT EVALUATION RESULTS")
print("=" * 65)

print(results_df.to_string(index=False))

print("\nSaved:", OUTPUT_PATH)
print("\nExperiment completed.")

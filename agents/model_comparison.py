
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


# ==================================================
# 1. PATHS AND CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "prediction_dataset.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = REPORT_DIR / "model_comparison.csv"
PREDICTIONS_PATH = REPORT_DIR / "model_comparison_predictions.csv"


# ==================================================
# 2. FEATURE DEFINITIONS
# ==================================================

MARKET_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "daily_return",
    "momentum_5",
    "sma_ratio",
    "volatility_5",
    "volume_change",
    "daily_range"
]

SENTIMENT_FEATURES = [
    "sentiment_score_lag1",
    "sentiment_count_lag1",
    "avg_confidence_lag1"
]

CATEGORICAL_FEATURES = ["ticker"]


# ==================================================
# 3. LOAD DATA
# ==================================================

print("=" * 60)
print("MODEL COMPARISON EXPERIMENT")
print("=" * 60)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Prediction dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("\nLoaded records:", len(df))
print("Available columns:", df.columns.tolist())


# ==================================================
# 4. VALIDATE AND CLEAN DATA
# ==================================================

required = [
    "date",
    "ticker",
    "target_return"
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
    utc=True
)

df["target_return"] = pd.to_numeric(
    df["target_return"],
    errors="coerce"
)

df = df.dropna(
    subset=["date", "ticker", "target_return"]
).copy()

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.sort_values("date").reset_index(drop=True)

if len(df) < 100:
    raise ValueError(
        "Not enough valid records for model comparison."
    )

market_features = [
    col for col in MARKET_FEATURES
    if col in df.columns
]

sentiment_features = [
    col for col in SENTIMENT_FEATURES
    if col in df.columns
]

if not market_features:
    raise ValueError("No market features found.")

print("\nValid records:", len(df))
print("Market features:", market_features)
print("Sentiment features:", sentiment_features)


# ==================================================
# 5. CHRONOLOGICAL TRAIN / TEST SPLIT
# ==================================================

unique_dates = sorted(df["date"].unique())

split_index = int(len(unique_dates) * 0.80)

if split_index <= 0 or split_index >= len(unique_dates):
    raise ValueError("Unable to create chronological split.")

split_date = unique_dates[split_index]

train = df[df["date"] < split_date].copy()
test = df[df["date"] >= split_date].copy()

print("\nTRAINING RECORDS:", len(train))
print("TESTING RECORDS:", len(test))
print("TEST START DATE:", split_date)

if train.empty or test.empty:
    raise ValueError("Training or testing dataset is empty.")


# ==================================================
# 6. BUILD MODEL PIPELINE
# ==================================================

def build_model(model_type, numeric_features):

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )

    if model_type == "Linear Regression":

        estimator = LinearRegression()

    elif model_type == "Random Forest":

        estimator = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        )

    else:
        raise ValueError("Unknown model type.")

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", estimator)
        ]
    )


# ==================================================
# 7. EVALUATION FUNCTION
# ==================================================

results = []
prediction_records = []


def evaluate_model(model_name, model_type, features):

    print(f"\nTraining: {model_name}")

    all_features = features + CATEGORICAL_FEATURES

    X_train = train[all_features]
    X_test = test[all_features]

    y_train = train["target_return"].to_numpy() * 100
    y_test = test["target_return"].to_numpy() * 100

    model = build_model(
        model_type,
        features
    )

    model.fit(X_train, y_train)

    predicted = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predicted
        )
    )

    directional_accuracy = np.mean(
        np.sign(y_test) == np.sign(predicted)
    ) * 100

    results.append({
        "model": model_name,
        "test_records": len(y_test),
        "MAE_Return_Percent": mae,
        "RMSE_Return_Percent": rmse,
        "Directional_Accuracy_Percent": directional_accuracy
    })

    for i in range(len(test)):

        prediction_records.append({
            "date": test.iloc[i]["date"],
            "ticker": test.iloc[i]["ticker"],
            "model": model_name,
            "actual_return_percent": y_test[i],
            "predicted_return_percent": predicted[i]
        })

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print(
        "Directional accuracy:",
        round(directional_accuracy, 2),
        "%"
    )


# ==================================================
# 8. ZERO RETURN BASELINE
# ==================================================

print("\nEvaluating zero-return baseline...")

y_test = test["target_return"].to_numpy() * 100

baseline_prediction = np.zeros(len(y_test))

baseline_mae = mean_absolute_error(
    y_test,
    baseline_prediction
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_prediction
    )
)

results.append({
    "model": "Zero Return Baseline",
    "test_records": len(y_test),
    "MAE_Return_Percent": baseline_mae,
    "RMSE_Return_Percent": baseline_rmse,
    "Directional_Accuracy_Percent": np.nan
})


# ==================================================
# 9. MARKET-ONLY MODELS
# ==================================================

evaluate_model(
    "Linear Regression - Market Only",
    "Linear Regression",
    market_features
)

evaluate_model(
    "Random Forest - Market Only",
    "Random Forest",
    market_features
)


# ==================================================
# 10. MARKET + SENTIMENT MODELS
# ==================================================

if sentiment_features:

    combined_features = (
        market_features + sentiment_features
    )

    evaluate_model(
        "Linear Regression - Market + Sentiment",
        "Linear Regression",
        combined_features
    )

    evaluate_model(
        "Random Forest - Market + Sentiment",
        "Random Forest",
        combined_features
    )

else:

    print(
        "\nSentiment features are missing. "
        "Skipping combined models."
    )


# ==================================================
# 11. SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "MAE_Return_Percent"
).reset_index(drop=True)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

predictions_df = pd.DataFrame(
    prediction_records
)

predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ==================================================
# 12. DISPLAY FINAL COMPARISON
# ==================================================

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(index=False)
)

print("\nBest model by MAE:")

print(
    results_df.iloc[0]["model"]
)

print("\nFiles saved:")
print(OUTPUT_PATH)
print(PREDICTIONS_PATH)

print("\nMODEL COMPARISON COMPLETED!")

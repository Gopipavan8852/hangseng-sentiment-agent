
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

MARKET_FILE = DATA_DIR / "market_data.csv"
SENTIMENT_FILE = DATA_DIR / "sentiment_results.csv"

REPORT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# TIMEZONE HANDLING
# --------------------------------------------------

def get_timezone(ticker):
    if str(ticker).endswith(".HK"):
        return "Asia/Hong_Kong"

    if str(ticker).endswith(".KS"):
        return "Asia/Seoul"

    return "UTC"


def local_date(value, ticker):
    if pd.isna(value):
        return pd.NaT

    try:
        timestamp = pd.to_datetime(value, utc=True)
        timestamp = timestamp.tz_convert(get_timezone(ticker))
        return timestamp.date()
    except Exception:
        return pd.NaT


# --------------------------------------------------
# LOAD MARKET DATA
# --------------------------------------------------

def load_market_data():

    print("\nLoading market data...")

    df = pd.read_csv(MARKET_FILE)
    df.columns = df.columns.str.strip().str.lower()

    required = [
        "date", "ticker", "open",
        "high", "low", "close", "volume"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing market columns: {missing}")

    df = df.reset_index(drop=True)

    df["ticker"] = df["ticker"].astype(str).str.strip()

    df["date"] = [
        local_date(d, t)
        for d, t in zip(df["date"], df["ticker"])
    ]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["date", "ticker", "close"]
    )

    df = df.sort_values(
        ["ticker", "date"]
    ).drop_duplicates(
        ["ticker", "date"],
        keep="last"
    )

    print("Market records:", len(df))
    print("Stocks:", df["ticker"].nunique())

    return df


# --------------------------------------------------
# LOAD SENTIMENT DATA
# --------------------------------------------------

def load_sentiment_data():

    print("\nLoading sentiment data...")

    df = pd.read_csv(SENTIMENT_FILE)
    df.columns = df.columns.str.strip().str.lower()

    required = ["ticker", "published_date", "sentiment"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing sentiment columns: {missing}")

    df["ticker"] = df["ticker"].astype(str).str.strip()

    df["date"] = [
        local_date(d, t)
        for d, t in zip(df["published_date"], df["ticker"])
    ]

    sentiment_map = {
        "very negative": -2,
        "negative": -1,
        "neutral": 0,
        "positive": 1,
        "very positive": 2
    }

    df["sentiment_score"] = (
        df["sentiment"]
        .astype(str)
        .str.lower()
        .str.strip()
        .map(sentiment_map)
    )

    df["confidence"] = pd.to_numeric(
        df.get("confidence", pd.Series(index=df.index, dtype=float)),
        errors="coerce"
    )

    df = df.dropna(
        subset=["ticker", "date", "sentiment_score"]
    )

    daily = (
        df.groupby(["ticker", "date"])
        .agg(
            sentiment_score=("sentiment_score", "mean"),
            sentiment_count=("sentiment_score", "count"),
            avg_confidence=("confidence", "mean")
        )
        .reset_index()
    )

    print("Valid sentiment records:", len(df))
    print("Daily sentiment groups:", len(daily))

    return daily


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

FEATURES = [
    "daily_return",
    "momentum_5",
    "sma_ratio",
    "volatility_5",
    "volume_change",
    "daily_range",
    "sentiment_score_lag1",
    "sentiment_count_lag1",
    "avg_confidence_lag1"
]


def prepare_dataset(market, sentiment):

    print("\nPreparing prediction features...")

    df = market.merge(
        sentiment,
        on=["ticker", "date"],
        how="left"
    )

    df = df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    for col in [
        "sentiment_score",
        "sentiment_count",
        "avg_confidence"
    ]:
        df[col] = df[col].fillna(0)

    grouped = df.groupby("ticker", sort=False)

    df["daily_return"] = grouped["close"].pct_change()

    df["momentum_5"] = grouped["close"].transform(
        lambda s: s.pct_change(5)
    )

    sma5 = grouped["close"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    df["sma_ratio"] = df["close"] / sma5 - 1

    df["volatility_5"] = grouped["daily_return"].transform(
        lambda s: s.rolling(5, min_periods=2).std()
    )

    df["volume_change"] = grouped["volume"].pct_change()

    df["daily_range"] = (
        (df["high"] - df["low"]) / df["close"]
    )

    # Use sentiment from the previous trading session.
    for col in [
        "sentiment_score",
        "sentiment_count",
        "avg_confidence"
    ]:
        df[col + "_lag1"] = (
            df.groupby("ticker")[col].shift(1)
        )

    # Predict next session's percentage return.
    df["target_return"] = (
        df.groupby("ticker")["close"].shift(-1)
        / df["close"] - 1
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    print("Prepared records:", len(df))

    return df


# --------------------------------------------------
# MODEL FEATURES
# --------------------------------------------------

def make_features(df, columns=None):

    if columns is None:
        columns = FEATURES

    numeric = df[columns].copy()

    ticker_encoded = pd.get_dummies(
        df["ticker"].astype(str),
        prefix="ticker",
        dtype=int
    )

    return pd.concat(
        [
            numeric.reset_index(drop=True),
            ticker_encoded.reset_index(drop=True)
        ],
        axis=1
    )


# --------------------------------------------------
# TRAIN AND EVALUATE
# --------------------------------------------------

def train_model(df):

    print("\nTraining percentage-return prediction model...")

    usable = df.dropna(
        subset=FEATURES + ["target_return"]
    ).copy()

    if len(usable) < 100:
        raise ValueError(
            f"Not enough usable records: {len(usable)}"
        )

    dates = sorted(usable["date"].unique())

    if len(dates) < 2:
        raise ValueError("Not enough distinct dates for testing.")

    split = int(len(dates) * 0.80)
    split = max(1, min(split, len(dates) - 1))

    cutoff = dates[split]

    train = usable[usable["date"] < cutoff].copy()
    test = usable[usable["date"] >= cutoff].copy()

    print("Training records:", len(train))
    print("Testing records:", len(test))
    print("Test start date:", cutoff)

    X_train = make_features(train)
    X_test = make_features(test)

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    y_train = train["target_return"]
    y_test = test["target_return"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Baseline: predict zero percentage return.
    baseline = np.zeros(len(test))

    model_mae = mean_absolute_error(y_test, predictions)
    model_rmse = np.sqrt(mean_squared_error(y_test, predictions))

    baseline_mae = mean_absolute_error(y_test, baseline)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline))

    actual_direction = np.sign(y_test.to_numpy())
    predicted_direction = np.sign(predictions)

    accuracy = np.mean(
        actual_direction == predicted_direction
    ) * 100

    metrics = pd.DataFrame([
        {
            "model": "Random Forest",
            "MAE_Return_Percent": model_mae * 100,
            "RMSE_Return_Percent": model_rmse * 100,
            "Directional_Accuracy_Percent": accuracy
        },
        {
            "model": "Zero Return Baseline",
            "MAE_Return_Percent": baseline_mae * 100,
            "RMSE_Return_Percent": baseline_rmse * 100,
            "Directional_Accuracy_Percent": 0.0
        }
    ])

    metrics.to_csv(
        REPORT_DIR / "model_metrics.csv",
        index=False
    )

    print("\nMODEL EVALUATION")
    print("=" * 65)
    print(metrics.to_string(index=False))

    # Per-stock test evaluation.
    test_results = test[
        ["date", "ticker", "close", "target_return"]
    ].copy()

    test_results["predicted_return"] = predictions

    test_results["predicted_next_close"] = (
        test_results["close"] * (1 + predictions)
    )

    test_results["actual_next_close"] = (
        test_results["close"] * (1 + test_results["target_return"])
    )

    test_results["absolute_error_return_percent"] = (
        (
            test_results["predicted_return"]
            - test_results["target_return"]
        ).abs() * 100
    )

    test_results.to_csv(
        REPORT_DIR / "model_test_predictions.csv",
        index=False
    )

    per_stock = (
        test_results.groupby("ticker")
        .agg(
            test_records=("target_return", "count"),
            actual_average_return=("target_return", "mean"),
            predicted_average_return=("predicted_return", "mean"),
            average_absolute_error=("absolute_error_return_percent", "mean")
        )
        .reset_index()
    )

    per_stock.to_csv(
        REPORT_DIR / "per_stock_evaluation.csv",
        index=False
    )

    print("\nPER-STOCK EVALUATION")
    print(per_stock.to_string(index=False))

    return model, X_train.columns.tolist()


# --------------------------------------------------
# PREDICT NEXT TRADING SESSION
# --------------------------------------------------

def predict_next_day(model, feature_columns, df):

    print("\nGenerating next-session predictions...")

    latest = (
        df.sort_values(["ticker", "date"])
        .groupby("ticker")
        .tail(1)
        .copy()
        .reset_index(drop=True)
    )

    latest = latest.dropna(subset=FEATURES).copy()

    if latest.empty:
        raise ValueError("No complete latest feature rows.")

    X_latest = make_features(latest)

    X_latest = X_latest.reindex(
        columns=feature_columns,
        fill_value=0
    )

    predicted_returns = model.predict(X_latest)

    latest["predicted_return_pct"] = predicted_returns * 100

    latest["predicted_next_close"] = (
        latest["close"] * (1 + predicted_returns)
    )

    latest["predicted_direction"] = np.where(
        predicted_returns > 0,
        "UP",
        np.where(predicted_returns < 0, "DOWN", "FLAT")
    )

    output = latest[[
        "date",
        "ticker",
        "close",
        "predicted_next_close",
        "predicted_return_pct",
        "predicted_direction",
        "sentiment_score_lag1",
        "sentiment_count_lag1",
        "avg_confidence_lag1"
    ]].copy()

    output = output.rename(columns={
        "date": "latest_price_date",
        "close": "current_close",
        "sentiment_score_lag1": "previous_session_sentiment",
        "sentiment_count_lag1": "previous_session_sentiment_count",
        "avg_confidence_lag1": "previous_session_avg_confidence"
    })

    output.to_csv(
        REPORT_DIR / "next_day_predictions.csv",
        index=False
    )

    print("\nNEXT TRADING SESSION PREDICTIONS")
    print("=" * 90)
    print(output.round(4).to_string(index=False))

    print("\nSaved: reports/next_day_predictions.csv")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 65)
    print("NEXT TRADING DAY PREDICTION AGENT")
    print("=" * 65)

    market = load_market_data()
    sentiment = load_sentiment_data()

    dataset = prepare_dataset(
        market,
        sentiment
    )

    dataset.to_csv(
        DATA_DIR / "prediction_dataset.csv",
        index=False
    )

    model, feature_columns = train_model(dataset)

    predict_next_day(
        model,
        feature_columns,
        dataset
    )

    print("\n" + "=" * 65)
    print("PREDICTION AGENT COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()

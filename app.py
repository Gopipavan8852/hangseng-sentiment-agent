
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"

st.set_page_config(
    page_title="Social Sentiment & Stock Prediction Agent",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Social Sentiment & Stock Prediction Agent")
st.caption("Hong Kong and South Korea | Social Media + News + Market Data")

st.warning(
    "Experimental research prototype — not investment advice. "
    "The current Random Forest model underperforms the zero-return "
    "baseline on MAE and RMSE. Predictions are uncertain."
)


# ==================================================
# 2. CSV LOADING FUNCTION
# ==================================================

@st.cache_data
def load_csv(file_path):

    path = Path(file_path)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except Exception as error:
        st.error(f"Error loading {path.name}: {error}")
        return pd.DataFrame()


market = load_csv(DATA_DIR / "market_data.csv")
sentiment = load_csv(DATA_DIR / "sentiment_results.csv")
news = load_csv(DATA_DIR / "news.csv")
social = load_csv(DATA_DIR / "social_data.csv")
combined = load_csv(DATA_DIR / "combined_data.csv")

metrics = load_csv(REPORT_DIR / "model_metrics.csv")
predictions = load_csv(REPORT_DIR / "next_day_predictions.csv")
stock_evaluation = load_csv(REPORT_DIR / "per_stock_evaluation.csv")


# ==================================================
# 3. MARKET DATA VALIDATION
# ==================================================

if market.empty:
    st.error("Market data is missing. Run stock_data_agent.py first.")
    st.stop()

required_columns = ["ticker", "date", "close"]

missing_columns = [
    column for column in required_columns
    if column not in market.columns
]

if missing_columns:
    st.error(f"Missing market data columns: {missing_columns}")
    st.write("Available columns:", market.columns.tolist())
    st.stop()


# ==================================================
# 4. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("📊 Dashboard Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Market Data",
        "Sentiment Analysis",
        "Stock Predictions",
        "Model Evaluation",
        "News & Social Data"
    ]
)


# ==================================================
# 5. OVERVIEW
# ==================================================

if page == "Overview":

    st.header("Project Overview")

    total_stocks = market["ticker"].nunique()
    total_market_records = len(market)
    total_sentiment = len(sentiment)
    total_news = len(news)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Stocks Tracked", total_stocks)
    col2.metric("Market Records", f"{total_market_records:,}")
    col3.metric("Sentiment Records", total_sentiment)
    col4.metric("News Records", total_news)

    # Latest stock prices

    st.subheader("Latest Stock Prices")

    latest_market = market.copy()

    # IMPORTANT: Handle mixed Hong Kong and Korea time zones.
    latest_market["date"] = pd.to_datetime(
        latest_market["date"],
        errors="coerce",
        utc=True
    )

    latest_market = latest_market.dropna(subset=["date"])

    latest_market = (
        latest_market
        .sort_values("date")
        .groupby("ticker")
        .tail(1)
        .sort_values("ticker")
    )

    display_columns = [
        column for column in [
            "ticker",
            "company",
            "date",
            "close",
            "volume"
        ]
        if column in latest_market.columns
    ]

    st.dataframe(
        latest_market[display_columns],
        width="stretch",
        hide_index=True
    )

    # Next trading day predictions

    st.subheader("Next Trading Day Predictions")

    if predictions.empty:

        st.info("Prediction results are not available.")

    else:

        st.dataframe(
            predictions,
            width="stretch",
            hide_index=True
        )


# ==================================================
# 6. MARKET DATA
# ==================================================

elif page == "Market Data":

    st.header("📉 Historical Market Data")

    tickers = sorted(
        market["ticker"].dropna().unique().tolist()
    )

    if not tickers:
        st.warning("No stock tickers found.")
        st.stop()

    selected_ticker = st.selectbox(
        "Select Stock",
        tickers
    )

    stock = market[
        market["ticker"] == selected_ticker
    ].copy()

    # Convert mixed timezone timestamps safely.
    stock["date"] = pd.to_datetime(
        stock["date"],
        errors="coerce",
        utc=True
    )

    stock = (
        stock.dropna(subset=["date"])
        .sort_values("date")
    )

    if stock.empty:
        st.warning("No valid market dates found for this stock.")
        st.stop()

    st.subheader(f"{selected_ticker} Closing Price")

    price_fig = px.line(
        stock,
        x="date",
        y="close",
        title=f"{selected_ticker} Historical Closing Price"
    )

    st.plotly_chart(
        price_fig,
        width="stretch"
    )

    if "volume" in stock.columns:

        st.subheader("Trading Volume")

        volume_fig = px.bar(
            stock.tail(60),
            x="date",
            y="volume",
            title="Recent Trading Volume"
        )

        st.plotly_chart(
            volume_fig,
            width="stretch"
        )

    st.subheader("Recent Market Records")

    st.dataframe(
        stock.tail(30),
        width="stretch",
        hide_index=True
    )


# ==================================================
# 7. SENTIMENT ANALYSIS
# ==================================================

elif page == "Sentiment Analysis":

    st.header("🧠 Multilingual Sentiment Analysis")

    if sentiment.empty:
        st.warning("Sentiment results are not available.")
        st.stop()

    st.metric(
        "Total Sentiment Records",
        len(sentiment)
    )

    if "sentiment" in sentiment.columns:

        sentiment_counts = (
            sentiment["sentiment"]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )

        sentiment_counts.columns = [
            "Sentiment",
            "Count"
        ]

        sentiment_fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count",
            title="Sentiment Distribution"
        )

        st.plotly_chart(
            sentiment_fig,
            width="stretch"
        )

    if (
        "ticker" in sentiment.columns
        and "sentiment" in sentiment.columns
    ):

        st.subheader("Sentiment by Stock")

        ticker_sentiment = (
            sentiment.groupby(
                ["ticker", "sentiment"]
            )
            .size()
            .reset_index(name="count")
        )

        ticker_fig = px.bar(
            ticker_sentiment,
            x="ticker",
            y="count",
            color="sentiment",
            barmode="group",
            title="Sentiment Comparison Across Stocks"
        )

        st.plotly_chart(
            ticker_fig,
            width="stretch"
        )

    st.subheader("Sentiment Records")

    st.dataframe(
        sentiment,
        width="stretch",
        hide_index=True
    )


# ==================================================
# 8. STOCK PREDICTIONS
# ==================================================

elif page == "Stock Predictions":

    st.header("🔮 Next Trading Day Predictions")

    st.caption(
        "Experimental model outputs. Predictions are uncertain "
        "and are not financial advice."
    )

    if predictions.empty:

        st.warning("Prediction results are not available.")

    else:

        st.dataframe(
            predictions,
            width="stretch",
            hide_index=True
        )

        ticker_col = next(
            (
                column for column in ["ticker", "Ticker"]
                if column in predictions.columns
            ),
            None
        )

        direction_col = next(
            (
                column for column in [
                    "predicted_direction",
                    "direction",
                    "prediction"
                ]
                if column in predictions.columns
            ),
            None
        )

        return_col = next(
            (
                column for column in [
                    "predicted_return_percent",
                    "predicted_return",
                    "predicted_return_pct"
                ]
                if column in predictions.columns
            ),
            None
        )

        if ticker_col and return_col:

            prediction_fig = px.bar(
                predictions,
                x=ticker_col,
                y=return_col,
                color=direction_col if direction_col else None,
                title="Predicted Percentage Return by Stock"
            )

            st.plotly_chart(
                prediction_fig,
                width="stretch"
            )


# ==================================================
# 9. MODEL EVALUATION
# ==================================================

elif page == "Model Evaluation":

    st.header("🤖 Machine Learning Model Evaluation")

    if metrics.empty:

        st.warning("Model evaluation metrics are not available.")

    else:

        st.subheader("Overall Model Metrics")

        st.dataframe(
            metrics,
            width="stretch",
            hide_index=True
        )

    if not stock_evaluation.empty:

        st.subheader("Per-Stock Evaluation")

        st.dataframe(
            stock_evaluation,
            width="stretch",
            hide_index=True
        )

    st.warning(
        "The current Random Forest model has higher MAE and RMSE "
        "than the zero-return baseline. Further improvement "
        "and validation are required."
    )


# ==================================================
# 10. NEWS AND SOCIAL DATA
# ==================================================

elif page == "News & Social Data":

    st.header("📰 Financial News and Social Media Data")

    if not news.empty:

        st.subheader("Financial News")

        news_columns = [
            column for column in [
                "market",
                "ticker",
                "company",
                "title",
                "published_date",
                "url"
            ]
            if column in news.columns
        ]

        st.dataframe(
            news[news_columns],
            width="stretch",
            hide_index=True
        )

    else:

        st.info("News data is not available.")

    if not combined.empty:

        st.subheader("Combined News and Social Records")

        st.dataframe(
            combined,
            width="stretch",
            hide_index=True
        )

    if not social.empty:

        st.subheader("Social Media Records")

        st.dataframe(
            social,
            width="stretch",
            hide_index=True
        )

    if news.empty and combined.empty and social.empty:

        st.warning("No news or social records were found.")


# ==================================================
# 11. FOOTER
# ==================================================

st.divider()

st.caption(
    "Hang Seng Social Sentiment & Price Prediction Agent | "
    "Python | Apify | Exa | Transformers | Scikit-learn | "
    "Pandas | Streamlit"
)

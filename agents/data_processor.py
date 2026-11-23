
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# Target stocks
STOCKS = {
    "0005.HK": {
        "market": "Hong Kong",
        "company": "HSBC Holdings",
        "keywords": ["HSBC", "滙豐", "汇丰"]
    },
    "0700.HK": {
        "market": "Hong Kong",
        "company": "Tencent",
        "keywords": ["Tencent", "騰訊", "腾讯"]
    },
    "9988.HK": {
        "market": "Hong Kong",
        "company": "Alibaba",
        "keywords": ["Alibaba", "阿里巴巴"]
    },
    "1810.HK": {
        "market": "Hong Kong",
        "company": "Xiaomi",
        "keywords": ["Xiaomi", "小米"]
    },
    "1299.HK": {
        "market": "Hong Kong",
        "company": "AIA Group",
        "keywords": ["AIA", "友邦"]
    },
    "005930.KS": {
        "market": "South Korea",
        "company": "Samsung Electronics",
        "keywords": ["Samsung", "삼성전자"]
    },
    "000660.KS": {
        "market": "South Korea",
        "company": "SK Hynix",
        "keywords": ["SK Hynix", "SK하이닉스", "하이닉스"]
    },
    "035420.KS": {
        "market": "South Korea",
        "company": "NAVER",
        "keywords": ["NAVER", "네이버"]
    },
    "035720.KS": {
        "market": "South Korea",
        "company": "Kakao",
        "keywords": ["Kakao", "카카오"]
    },
    "005380.KS": {
        "market": "South Korea",
        "company": "Hyundai Motor",
        "keywords": ["Hyundai", "현대자동차", "현대차"]
    }
}

TICKER_ALIASES = {
    "0005": "0005.HK",
    "0005.HK": "0005.HK",
    "0700": "0700.HK",
    "0700.HK": "0700.HK",
    "9988": "9988.HK",
    "9988.HK": "9988.HK",
    "1810": "1810.HK",
    "1810.HK": "1810.HK",
    "1299": "1299.HK",
    "1299.HK": "1299.HK",
    "005930": "005930.KS",
    "005930.KS": "005930.KS",
    "000660": "000660.KS",
    "000660.KS": "000660.KS",
    "035420": "035420.KS",
    "035420.KS": "035420.KS",
    "035720": "035720.KS",
    "035720.KS": "035720.KS",
    "005380": "005380.KS",
    "005380.KS": "005380.KS"
}

def read_csv(filename):
    path = DATA_DIR / filename

    if not path.exists() or path.stat().st_size == 0:
        print("Skipping empty or missing file:", filename)
        return pd.DataFrame()

    for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    return pd.DataFrame()


def find_column(df, possible_names):
    lookup = {
        str(col).lower().strip(): col
        for col in df.columns
    }

    for name in possible_names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def clean_value(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def identify_stock(text, explicit_ticker=""):
    ticker = clean_value(explicit_ticker).upper()

    if ticker in TICKER_ALIASES:
        return TICKER_ALIASES[ticker]

    text = str(text).lower()

    for stock_ticker, details in STOCKS.items():
        for keyword in details["keywords"]:
            if keyword.lower() in text:
                return stock_ticker

    return None


def process_file(filename, source, text_type):
    df = read_csv(filename)

    if df.empty:
        return []

    text_column = find_column(
        df,
        ["text", "content", "own_text", "news_summary",
         "title", "description", "message", "body"]
    )

    if text_column is None:
        print("No text column found:", filename)
        return []

    ticker_column = find_column(
        df, ["ticker", "symbol", "stock_code", "stockcode", "code"]
    )

    date_column = find_column(
        df, ["date", "created_at", "published_date",
             "publishedAt", "timestamp", "time"]
    )

    results = []

    for _, row in df.iterrows():
        text = clean_value(row[text_column])

        if not text:
            continue

        explicit_ticker = (
            clean_value(row[ticker_column])
            if ticker_column else ""
        )

        ticker = identify_stock(text, explicit_ticker)

        if not ticker:
            continue

        details = STOCKS[ticker]

        published_date = (
            row[date_column] if date_column else None
        )

        results.append({
            "market": details["market"],
            "ticker": ticker,
            "company": details["company"],
            "source": source,
            "text": text,
            "text_type": text_type,
            "published_date": published_date
        })

    print(
        f"{filename}: {len(results)} relevant records "
        f"out of {len(df)}"
    )

    return results


def main():
    print("\nSTARTING DATA PROCESSING\n")

    datasets = []

    # Hong Kong and South Korean social sources
    datasets += process_file(
        "lihkg.csv", "LIHKG", "headline"
    )

    datasets += process_file(
        "telegram.csv", "Telegram", "message"
    )

    datasets += process_file(
        "naver.csv", "Naver", "post"
    )

    datasets += process_file(
        "clien.csv", "Clien", "headline"
    )

    # News dataset
    datasets += process_file(
        "news.csv", "Exa News", "news_summary"
    )

    if not datasets:
        print("\nNo matching stock records found.")
        print("Check your source columns and stock keywords.")
        return

    combined = pd.DataFrame(datasets)

    combined["published_date"] = pd.to_datetime(
        combined["published_date"],
        errors="coerce",
        utc=True
    )

    combined["text"] = (
        combined["text"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    combined = combined[combined["text"] != ""]

    combined = combined.drop_duplicates(
        subset=["ticker", "source", "text"]
    )

    # Save all sources together
    combined_path = DATA_DIR / "combined_data.csv"

    combined.to_csv(
        combined_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Save only social-media sources
    social = combined[
        combined["source"] != "Exa News"
    ]

    social_path = DATA_DIR / "social_data.csv"

    social.to_csv(
        social_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Save source and ticker counts
    summary = (
        combined.groupby(["market", "ticker", "source"])
        .size()
        .reset_index(name="record_count")
    )

    summary.to_csv(
        REPORT_DIR / "processing_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nPROCESSING COMPLETED")
    print("Total relevant records:", len(combined))
    print("Social records:", len(social))
    print("Combined file:", combined_path)
    print("Social file:", social_path)

    print("\nRecords by source:")
    print(combined["source"].value_counts().to_string())

    print("\nRecords by ticker:")
    print(combined["ticker"].value_counts().to_string())


if __name__ == "__main__":
    main()


import yfinance as yf
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

STOCKS = {
    "0005.HK": "HSBC Holdings",
    "0700.HK": "Tencent",
    "9988.HK": "Alibaba",
    "1810.HK": "Xiaomi",
    "1299.HK": "AIA Group",
    "005930.KS": "Samsung Electronics",
    "000660.KS": "SK Hynix",
    "035420.KS": "NAVER",
    "035720.KS": "Kakao",
    "005380.KS": "Hyundai Motor"
}

def main():
    all_data = []

    print("\nDOWNLOADING HISTORICAL STOCK DATA\n")

    for ticker, company in STOCKS.items():
        print(f"Downloading {company} ({ticker})...")

        try:
            stock = yf.Ticker(ticker)

            df = stock.history(
                period="2y",
                interval="1d",
                auto_adjust=True
            )

            if df.empty:
                print("No data returned:", ticker)
                continue

            df = df.reset_index()

            df["ticker"] = ticker
            df["company"] = company

            df = df.rename(columns={
                "Date": "date",
                "Datetime": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })

            required = [
                "date", "ticker", "company",
                "open", "high", "low", "close", "volume"
            ]

            df = df[required]

            all_data.append(df)

            print("Downloaded records:", len(df))

        except Exception as error:
            print("Download failed:", ticker, error)

    if not all_data:
        print("\nNo stock data downloaded.")
        return

    final_df = pd.concat(all_data, ignore_index=True)

    output_path = DATA_DIR / "market_data.csv"

    final_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDOWNLOAD COMPLETED")
    print("Total price records:", len(final_df))
    print("Stocks downloaded:", final_df["ticker"].nunique())
    print("Saved to:", output_path)

    print("\nRecords per stock:")
    print(final_df.groupby("ticker").size().to_string())


if __name__ == "__main__":
    main()

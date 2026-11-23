
import pandas as pd
import torch

from pathlib import Path
from transformers import pipeline

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"

INPUT_FILE = DATA_DIR / "combined_data.csv"
OUTPUT_FILE = DATA_DIR / "sentiment_results.csv"

MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"

def main():

    print("\nMULTILINGUAL SENTIMENT ANALYSIS")
    print("=" * 55)

    if not INPUT_FILE.exists():
        print("combined_data.csv not found!")
        return

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    if df.empty:
        print("No records available.")
        return

    print("Records loaded:", len(df))
    print("\nLoading multilingual model...")
    print("First run may take some time to download the model.")

    device = 0 if torch.cuda.is_available() else -1

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=device
    )

    results = []

    for index, row in df.iterrows():

        text = str(row["text"]).strip()

        if not text:
            continue

        try:
            prediction = sentiment_pipeline(
                text[:1500],
                truncation=True,
                max_length=512
            )[0]

            raw_label = str(prediction["label"]).upper()

            label_mapping = {
                "VERY NEGATIVE": "Very Negative",
                "NEGATIVE": "Negative",
                "NEUTRAL": "Neutral",
                "POSITIVE": "Positive",
                "VERY POSITIVE": "Very Positive"
            }

            label = label_mapping.get(
                raw_label,
                raw_label.title()
            )

            results.append({
                "market": row["market"],
                "ticker": row["ticker"],
                "company": row["company"],
                "source": row["source"],
                "text": text,
                "text_type": row.get("text_type", "unknown"),
                "published_date": row["published_date"],
                "sentiment": label,
                "confidence": float(prediction["score"])
            })

            print(
                f"{index + 1}/{len(df)} | "
                f"{row['ticker']} | {label}"
            )

        except Exception as error:
            print(f"Failed record {index + 1}: {error}")

    if not results:
        print("No sentiment results generated.")
        return

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # Sentiment summary for each stock
    summary = (
        result_df.groupby(["market", "ticker", "company", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    summary_path = REPORT_DIR / "sentiment_summary.csv"

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 55)
    print("SENTIMENT ANALYSIS COMPLETED")
    print("=" * 55)

    print("Analyzed records:", len(result_df))
    print("Sentiment results:", OUTPUT_FILE)
    print("Summary report:", summary_path)

    print("\nSentiment distribution:")
    print(result_df["sentiment"].value_counts().to_string())

    print("\nStock sentiment summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

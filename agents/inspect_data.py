
import pandas as pd
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

TEXT_COLUMNS = [
    "text", "content", "body", "message",
    "post_text", "news_summary", "description"
]

def load_csv(file_path):
    """Read CSV safely with common encodings."""
    for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read CSV encoding")


def audit_csv(file_path):
    print("\n" + "=" * 65)
    print("FILE:", file_path.name)
    print("=" * 65)

    try:
        df = load_csv(file_path)

        print("Total records:", len(df))
        print("Total columns:", len(df.columns))
        print("Columns:", list(df.columns))

        print("\nMissing values:")
        print(df.isnull().sum().to_string())

        print("\nDuplicate rows:", df.duplicated().sum())

        # Detect available text columns
        available_text = [
            col for col in df.columns
            if col.lower().strip() in TEXT_COLUMNS
        ]

        print("\nText columns found:", available_text)

        if available_text:
            text_col = available_text[0]
            valid_text = df[text_col].fillna("").astype(str).str.strip()
            valid_text = valid_text[valid_text != ""]

            print("Records with usable text:", len(valid_text))
            print("Sample text:")
            print(valid_text.head(2).to_string(index=False))
        else:
            print("WARNING: No recognized text column found.")

        return {
            "filename": file_path.name,
            "records": len(df),
            "columns": len(df.columns),
            "duplicate_rows": int(df.duplicated().sum()),
            "text_columns": ", ".join(available_text),
            "records_with_text": (
                len(valid_text) if available_text else 0
            ),
            "status": "Success"
        }

    except Exception as error:
        print("ERROR:", error)

        return {
            "filename": file_path.name,
            "records": 0,
            "columns": 0,
            "duplicate_rows": 0,
            "text_columns": "",
            "records_with_text": 0,
            "status": str(error)
        }


def main():
    csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in:", DATA_DIR)
        return

    print("HANG SENG PROJECT - DATA QUALITY AUDIT")
    print("CSV files found:", len(csv_files))

    results = []

    for file_path in csv_files:
        results.append(audit_csv(file_path))

    report = pd.DataFrame(results)

    report_path = REPORT_DIR / "data_quality_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    print(report.to_string(index=False))

    print("\nReport saved to:", report_path)


if __name__ == "__main__":
    main()

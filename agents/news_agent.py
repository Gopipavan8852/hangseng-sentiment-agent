
import os
import pandas as pd

from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
from exa_py import Exa


# --------------------------------
# 1. Load API credentials
# --------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("EXA_API_KEY")

if not api_key:
    raise ValueError("EXA_API_KEY not found in .env file")

# Connect to Exa
exa = Exa(api_key=api_key)


# --------------------------------
# 2. Define our 10 stocks
# --------------------------------

stocks = [
    {"market": "Hong Kong", "ticker": "0005.HK", "company": "HSBC Holdings"},
    {"market": "Hong Kong", "ticker": "0700.HK", "company": "Tencent"},
    {"market": "Hong Kong", "ticker": "9988.HK", "company": "Alibaba"},
    {"market": "Hong Kong", "ticker": "1810.HK", "company": "Xiaomi"},
    {"market": "Hong Kong", "ticker": "1299.HK", "company": "AIA Group"},
    {"market": "South Korea", "ticker": "005930.KS", "company": "Samsung Electronics"},
    {"market": "South Korea", "ticker": "000660.KS", "company": "SK Hynix"},
    {"market": "South Korea", "ticker": "035420.KS", "company": "NAVER"},
    {"market": "South Korea", "ticker": "035720.KS", "company": "Kakao"},
    {"market": "South Korea", "ticker": "005380.KS", "company": "Hyundai Motor"}
]


# --------------------------------
# 3. Calculate date range
# --------------------------------

today = datetime.now(timezone.utc).date()
start_date = (today - timedelta(days=10)).isoformat()

print("Collecting news from:", start_date)
print("To:", today.isoformat())


# --------------------------------
# 4. Collect financial news
# --------------------------------

all_news = []

for stock in stocks:

    company = stock["company"]
    ticker = stock["ticker"]
    market = stock["market"]

    print(f"\nSearching news for {company}...")

    query = (
        f"Latest financial news, company announcements "
        f"and business developments about {company} "
        f"({ticker}) in {market}"
    )

    try:

        results = exa.search(
            query,
            type="auto",
            category="news",
            num_results=3,
            start_published_date=start_date,
            contents={"highlights": True}
        )

        articles = results.results

        for article in articles:

            highlights = getattr(article, "highlights", None) or []

            # Clean summary: remove extra spaces and line breaks
            summary_text = " ".join(
                str(part) for part in highlights
            )
            summary_text = " ".join(summary_text.split())

            news_item = {
                "market": market,
                "ticker": ticker,
                "company": company,
                "title": article.title or "",
                "url": article.url or "",
                "published_date": article.published_date or "",
                "news_summary": summary_text,
                "collected_at_utc": datetime.now(timezone.utc).isoformat()
            }

            all_news.append(news_item)

        print(f"Collected {len(articles)} articles for {company}")

    except Exception as error:

        print(f"Error collecting news for {company}: {error}")


# --------------------------------
# 5. Save news into CSV
# --------------------------------

output_file = PROJECT_ROOT / "data" / "news.csv"

# Ensure data folder exists
output_file.parent.mkdir(parents=True, exist_ok=True)

if all_news:

    df = pd.DataFrame(all_news)

    df.drop_duplicates(
        subset=["ticker", "url"],
        inplace=True
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n-----------------------------")
    print("NEWS COLLECTION COMPLETED!")
    print("-----------------------------")

    print("Total articles:", len(df))
    print("Saved location:", output_file)

    print("\nArticles per company:")
    print(df.groupby("company").size())

else:

    print("\nNo news articles collected.")

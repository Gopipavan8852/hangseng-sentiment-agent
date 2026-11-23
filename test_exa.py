
import os
from dotenv import load_dotenv
from exa_py import Exa

# Load API credentials
load_dotenv()

api_key = os.getenv("EXA_API_KEY")

if not api_key:
    print("ERROR: Exa API key not found!")
else:
    try:
        # Connect to Exa
        exa = Exa(api_key=api_key)

        print("Connecting to Exa...")

        # Search for financial news
        results = exa.search(
            "Latest financial news about Hong Kong stock market",
            type="auto",
            num_results=3,
            contents={"highlights": True}
        )

        print("\nSUCCESS! Exa connected.")

        for article in results.results:
            print("\nTitle:", article.title)
            print("URL:", article.url)
            print("Published:", getattr(article, "published_date", None))

            highlights = getattr(article, "highlights", None)

            if highlights:
                print("News summary:", highlights[0])

    except Exception as error:
        print("Exa connection failed:", error)

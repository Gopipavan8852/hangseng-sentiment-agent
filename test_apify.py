
import os
from dotenv import load_dotenv
from apify_client import ApifyClient

# Load environment variables
load_dotenv()

api_token = os.getenv("APIFY_API_TOKEN")

if not api_token:
    print("ERROR: Apify token not found!")

else:
    try:
        # Connect to Apify
        client = ApifyClient(api_token)

        # Get account information
        account = client.user().get()

        if account:
            print("SUCCESS! Apify connected.")
            print("Username:", account.username)
            print("Account ID:", account.id)

        else:
            print("Account information not available.")

    except Exception as error:
        print("Apify connection failed:", error)

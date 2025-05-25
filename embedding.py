# src/embedding.py

import os
import sys
import json
import csv
import requests
from glob import glob
from tenacity import (
    retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
)
from openai.error import RateLimitError, APIError, Timeout, APIConnectionError
import openai
from dotenv import load_dotenv

load_dotenv()

# Read environment variables
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002")

if not (OPENAI_API_KEY and AZURE_ENDPOINT and AZURE_VERSION and AZURE_EMBED_DEPLOYMENT):
    raise ValueError("Missing Azure OpenAI environment variables for embeddings.")

openai.api_type = "azure"
openai.api_key = OPENAI_API_KEY
openai.api_base = AZURE_ENDPOINT
openai.api_version = AZURE_VERSION

FOLDER_PATH = os.environ.get("FOLDER_PATH", "data")       # Where your JSON files are
OUTPUT_CSV = os.path.join(FOLDER_PATH, "attributes_enriched.csv")
MAX_CHARS = int(os.environ.get("MAX_CHARS", 3000))

def check_connectivity():
    """Test a simple embedding call to ensure the endpoint is reachable."""
    url = f"{AZURE_ENDPOINT}openai/deployments/{AZURE_EMBED_DEPLOYMENT}/embeddings?api-version={AZURE_VERSION}"
    headers = {
        "api-key": OPENAI_API_KEY,
        "Content-Type": "application/json"
    }
    data = {"input": "test"}
    try:
        r = requests.post(url, headers=headers, json=data, timeout=10)
        print(f"[INFO] Connectivity check => {r.status_code}")
        if r.status_code not in (200, 201):
            print(f"[ERROR] Endpoint returned {r.status_code}: {r.text}")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Cannot reach endpoint: {str(e)}")
        sys.exit(1)

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((RateLimitError, APIError, Timeout, APIConnectionError))
)
def get_embedding_with_backoff(text: str):
    """Call the Azure OpenAI Embeddings endpoint with retries."""
    response = openai.Embedding.create(
        input=text,
        engine=AZURE_EMBED_DEPLOYMENT
    )
    return response["data"][0]["embedding"]

def main():
    check_connectivity()
    os.makedirs(FOLDER_PATH, exist_ok=True)

    # For demonstration, we assume you have a single extracted JSON: "output.json"
    # If you have multiple JSONs, you can glob them similarly to the original code.
    input_json = os.path.join(FOLDER_PATH, "output.json")
    if not os.path.exists(input_json):
        print(f"No JSON found at {input_json}. Exiting.")
        return

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "model_name", "entity_name", "attribute_name", "embedding"
        ])

        with open(input_json, 'r', encoding='utf-8') as f:
            records = json.load(f)

        # Example: Suppose each record has the structure:
        # {
        #   "Model": "...",
        #   "Entity": {
        #       "ENTITY NAME": "...",
        #       "Raw_Section_Text": "..."
        #   }
        # }
        # We want to parse out potential attributes from `Raw_Section_Text`.
        # For demonstration, let's assume each line in `Raw_Section_Text` might represent an attribute.
        for record in records:
            model_name = record.get("Model", "")
            entity_obj = record.get("Entity", {})
            entity_name = entity_obj.get("ENTITY NAME", "")
            raw_text = entity_obj.get("Raw_Section_Text", "")

            # Naive splitting for demonstration
            lines = raw_text.split("\n")
            for line in lines:
                line = line.strip()
                if not line or "Entity Name:" in line:
                    continue
                text_to_embed = line
                if len(text_to_embed) > MAX_CHARS:
                    # chunk if needed (omitted for brevity)
                    text_to_embed = text_to_embed[:MAX_CHARS]

                try:
                    embedding = get_embedding_with_backoff(text_to_embed)
                except Exception as e:
                    print(f"[ERROR] Embedding failed: {e}")
                    embedding = []

                # Write row
                writer.writerow([
                    model_name,
                    entity_name,
                    line,  # treat the line as the 'attribute_name'
                    json.dumps(embedding)
                ])

    print(f"Embedding complete. CSV saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

# src/data_extraction.py

import os
import json
import re
from PyPDF2 import PdfReader
import tiktoken
from dotenv import load_dotenv

# Load environment variables from .env (if you’re using that approach)
load_dotenv()

# Example environment variable usage
PDF_PATH = os.environ.get("PDF_PATH", "Data_Dictionary.pdf")  # path to your PDF
OUTPUT_JSON = os.environ.get("OUTPUT_JSON", "output.json")
TOKEN_THRESHOLD = int(os.environ.get("TOKEN_THRESHOLD", 7000))

token_encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Counts tokens using the cl100k_base encoding."""
    tokens = token_encoder.encode(text)
    return len(tokens)

def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF file not found at: {PDF_PATH}")

    print(f"Reading PDF from: {PDF_PATH}")
    reader = PdfReader(PDF_PATH)
    pages_text = []

    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        pages_text.append(text)

    full_text = "\n<PAGE_BREAK>\n".join(pages_text)
    print(f"PDF contains {len(pages_text)} pages.")

    # Example: Try to find model name
    model_match = re.search(r"Model\s*:\s*([^\n]+)", full_text, flags=re.IGNORECASE)
    if model_match:
        model_name = model_match.group(1).strip()
    else:
        model_name = "Not found"

    # Regex to detect "Entity Name:" or similar headings
    entity_pattern = re.compile(r"^(?:Entity\s*Name|ENTITY\s*NAME)\s*:\s*", flags=re.IGNORECASE | re.MULTILINE)
    entity_positions = [m.start() for m in entity_pattern.finditer(full_text)]
    if not entity_positions:
        raise ValueError("No entities found in PDF. Check PDF format or regex pattern.")
    entity_positions.append(len(full_text))

    # Simplistic segmentation: slice text into sections based on entity start indexes
    results = []
    for i in range(len(entity_positions) - 1):
        start_idx = entity_positions[i]
        end_idx = entity_positions[i + 1]
        section_text = full_text[start_idx:end_idx].strip()

        # Example parse: Extract the "Entity Name"
        match_entity = re.search(r"(?:Entity\s*Name|ENTITY\s*NAME)\s*:\s*(.*)", section_text, re.IGNORECASE)
        entity_name = match_entity.group(1).split("\n")[0].strip() if match_entity else f"Entity_{i+1}"

        # You can do more refined parsing for table name, attribute definitions, etc. 
        # For now we store raw text.
        results.append({
            "Model": model_name,
            "Entity": {
                "ENTITY NAME": entity_name,
                "Raw_Section_Text": section_text
            }
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Extraction completed. JSON written to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()

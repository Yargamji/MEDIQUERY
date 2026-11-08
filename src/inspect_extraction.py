import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

json_path = PROJECT_ROOT / "data" / "processed" / "documents.json"

with open(json_path, "r", encoding="utf-8") as file:
    documents = json.load(file)

for item in documents:

    if len(item["text"]) > 500:

        print("DOCUMENT:", item["document"])
        print("PAGE:", item["page"])
        print("\nEXTRACTED TEXT:\n")

        print(item["text"][:1500])

        break
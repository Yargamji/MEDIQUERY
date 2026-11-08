import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

json_path = PROJECT_ROOT / "data" / "processed" / "documents.json"

with open(json_path, "r", encoding="utf-8") as file:
    documents = json.load(file)

keywords = [
    "paracetamol",
    "amoxicillin",
    "artemether"
]

for keyword in keywords:

    print("\n" + "=" * 60)
    print(f"SEARCHING FOR: {keyword.upper()}")
    print("=" * 60)

    matches = [
        item for item in documents
        if keyword.lower() in item["text"].lower()
    ]

    for item in matches[:2]:

        print(f"\nPage: {item['page']}")
        print(item["text"][:2000])
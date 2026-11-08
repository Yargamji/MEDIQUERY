import json
import re
from pathlib import Path
from pypdf import PdfReader


# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input and output directories
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Output JSON file
OUTPUT_FILE = PROCESSED_DIR / "documents.json"


def clean_text(text):
    """
    Perform basic text cleaning while preserving
    paragraph structure and meaningful content.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Join words split across lines by hyphens
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf(pdf_path):
    """
    Extract text from a PDF page by page.

    Each extracted page retains its document name
    and original page number.
    """

    print(f"\nProcessing: {pdf_path.name}")

    reader = PdfReader(str(pdf_path))

    extracted_pages = []
    empty_pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            raw_text = page.extract_text() or ""
            cleaned_text = clean_text(raw_text)

            if len(cleaned_text) < 50:
                empty_pages.append(page_number)

            extracted_pages.append({
                "document": pdf_path.name,
                "page": page_number,
                "text": cleaned_text
            })

        except Exception as error:

            print(
                f"Warning: Could not extract page "
                f"{page_number}: {error}"
            )

            empty_pages.append(page_number)

            extracted_pages.append({
                "document": pdf_path.name,
                "page": page_number,
                "text": ""
            })

    # Document-level extraction report
    total_characters = sum(
        len(item["text"]) for item in extracted_pages
    )

    report = {
        "document": pdf_path.name,
        "total_pages": len(reader.pages),
        "extracted_pages": len(extracted_pages),
        "pages_with_little_or_no_text": empty_pages,
        "total_characters": total_characters
    }

    print(f"Total pages: {report['total_pages']}")
    print(f"Extracted pages: {report['extracted_pages']}")
    print(f"Pages with little/no text: {len(empty_pages)}")
    print(f"Total extracted characters: {total_characters:,}")

    return extracted_pages, report


def main():

    # Create output directory if it doesn't exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Locate all PDF documents
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF documents found in {RAW_DIR}"
        )

    all_documents = []
    extraction_reports = []

    print("=" * 60)
    print("NIGERIAN PUBLIC HEALTH DOCUMENT INGESTION")
    print("=" * 60)

    for pdf_path in pdf_files:

        pages, report = extract_pdf(pdf_path)

        all_documents.extend(pages)
        extraction_reports.append(report)

    # Save extracted documents
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(
            all_documents,
            file,
            ensure_ascii=False,
            indent=2
        )

    # Display final summary
    print("\n" + "=" * 60)
    print("INGESTION COMPLETED")
    print("=" * 60)

    print(f"Documents processed: {len(pdf_files)}")
    print(f"Total pages extracted: {len(all_documents)}")
    print(f"JSON output: {OUTPUT_FILE}")

    print("\nEXTRACTION SUMMARY")

    for report in extraction_reports:

        print(f"\n{report['document']}")
        print(f"Pages: {report['total_pages']}")
        print(f"Characters: {report['total_characters']:,}")
        print(
            f"Pages needing inspection: "
            f"{report['pages_with_little_or_no_text']}"
        )


if __name__ == "__main__":
    main()
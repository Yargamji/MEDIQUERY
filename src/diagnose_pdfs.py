from pathlib import Path
import fitz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def diagnose_pdf(pdf_path):

    print("\n" + "=" * 65)
    print(f"DOCUMENT: {pdf_path.name}")
    print("=" * 65)

    doc = fitz.open(pdf_path)

    total_characters = 0
    pages_with_text = 0

    print(f"Total pages: {len(doc)}")

    for page_number, page in enumerate(doc, start=1):

        text = page.get_text("text")

        characters = len(text.strip())
        total_characters += characters

        if characters > 50:
            pages_with_text += 1

        if page_number <= 5:
            print(
                f"Page {page_number}: "
                f"{characters} characters"
            )

    print("\nSUMMARY")
    print(f"Pages containing text: {pages_with_text}")
    print(f"Total extracted characters: {total_characters:,}")

    doc.close()


def main():

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:
        diagnose_pdf(pdf_path)


if __name__ == "__main__":
    main()
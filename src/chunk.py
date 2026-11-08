import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "documents.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"

CHUNK_SIZE = 800
OVERLAP = 100


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def get_section(page_number):
    """Assign the verified document-level section."""

    if page_number <= 9:
        return "Front Matter and Contents"

    elif page_number <= 50:
        return "Main Adult List"

    elif page_number <= 66:
        return "Primary Health Care List - Health Centres"

    elif page_number <= 72:
        return "Primary Health Care List - Primary Health Care Clinic"

    elif page_number <= 75:
        return "Primary Health Care List - Health Posts"

    elif page_number <= 85:
        return (
            "List of Products for Patent and "
            "Proprietary Medicine Vendors"
        )

    return "Appendices and Other Sections"


def canonicalize_heading(title):
    """
    Correct known OCR variations in subsection metadata.
    Original extracted text is not modified.
    """

    aliases = {
        "antbelmintbics": "Intestinal Anthelmintics",
        "intestinal antbelmintbics": "Intestinal Anthelmintics",
        "anthelminthics": "Anthelmintics",
        "antibacterial medicines": "Antibacterial Medicines",
        "antihypertensive medicines": "Antihypertensive Medicines",
        "oral rehydration": "Oral Rehydration",
    }

    normalized = normalize_text(title).lower()

    return aliases.get(normalized, title.title())


def detect_heading(line):
    """
    Identify subsection headings while avoiding false
    detection of medicine strengths and numerical expressions.

    Handles OCR errors such as S.2 and S.3 being read
    instead of 5.2 and 5.3.
    """

    line = normalize_text(line)

    if not line:
        return None

    # OCR correction for heading detection only.
    # The original line remains unchanged in the chunk text.
    heading_line = re.sub(
        r"^[sS](?=\.\s*\d)",
        "5",
        line
    )

    numbered_heading = re.match(
        r"^(\d{1,2})\.(\d{1,2})?(?:\.(\d{1,2}))?\.?\s+(.+)$",
        heading_line
        )

    if numbered_heading:

        main_number = int(numbered_heading.group(1))
        heading_title = numbered_heading.group(4).strip()

        if not heading_title:
            return None

        first_word = heading_title.split()[0].lower()

        non_heading_units = {
            "mg", "mcg", "g", "ml", "mm", "cm",
            "units", "unit", "kg", "%"
        }

        # Reject short OCR artefacts such as "6 G."
        # and medicine measurement expressions.
        if (
            1 <= main_number <= 31
            and len(first_word) >= 3
            and first_word.isalpha()
            and first_word not in non_heading_units
            and heading_title[0].isalpha()
        ):
            return canonicalize_heading(heading_title)

    known_headings = {

        "ANTIHYPERTENSIVE MEDICINES": "Antihypertensive Medicines",
        "ANTI-INFECTIVE MEDICINES": "Anti-infective Medicines",
        "ANTIARRHYTHMIC MEDICINES": "Anti-arrhythmic Medicines",
        "ANTI-ARRHYTHMIC MEDICINES": "Anti-arrhythmic Medicines",
        "ANTI-ANGINAL MEDICINES": "Anti-anginal Medicines",
        "MEDICINES USED IN HEART FAILURE": "Medicines Used in Heart Failure",
        "ANTI-THROMBOTIC MEDICINES": "Anti-thrombotic Medicines",
        "ANALGESICS": "Analgesics",
        "ANAESTHETICS (LOCAL)": "Anaesthetics (Local)",
        "ANTIDOTES": "Antidotes",
        "ANTICONVULSANTS": "Anticonvulsants",
        "ANTIALLERGICS": "Antiallergics",
        "ANTIBACTERIAL MEDICINES": "Antibacterial Medicines",
        "ANTIMALARIAL MEDICINES": "Antimalarial Medicines",
        "ANTHELMINTICS": "Anthelmintics",
        "ANTISEPTIC AND DISINFECTANTS": "Antiseptics and Disinfectants",
        "DIAGNOSTIC AGENTS": "Diagnostic Agents",
        "DRESSINGS": "Dressings",
        "EYE DROPS": "Eye Drops",
        "GASTROINTESTINAL MEDICINES": "Gastrointestinal Medicines",
        "HORMONES AND SYNTHETIC SUBSTITUTES": "Hormones and Synthetic Substitutes",
        "IMMUNOLOGICALS": "Immunologicals",
    }

    return known_headings.get(line.upper())


def split_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split a text segment into overlapping chunks."""

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        if end < len(text):

            boundary = text.rfind(". ", start, end)

            if boundary > start + chunk_size * 0.5:
                end = boundary + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def process_page(page, current_subsection, chunk_id):
    """
    Split a page into subsection-aware segments.

    Pages 1–9 are chunked without subsection detection.
    Pages 10 onward retain heading-aware chunking.
    """

    text = page["text"]
    page_number = int(page["page"])
    section = get_section(page_number)

    if page_number <= 9:

        current_subsection = None
        segments = [(None, text)]

    else:

        lines = text.splitlines()

        segments = []
        current_lines = []

        for line in lines:

            heading = detect_heading(line)

            if heading:

                # Save preceding text under its previous subsection.
                if current_lines:

                    segment_text = "\n".join(current_lines).strip()

                    if segment_text:
                        segments.append(
                            (current_subsection, segment_text)
                        )

                # Start the newly detected subsection.
                current_subsection = heading
                current_lines = [line]

            else:
                current_lines.append(line)

        # Save the final segment on the page.
        if current_lines:

            segment_text = "\n".join(current_lines).strip()

            if segment_text:
                segments.append(
                    (current_subsection, segment_text)
                )

    page_chunks = []

    for subsection, segment_text in segments:

        text_chunks = split_text(segment_text)

        for chunk_number, chunk_text in enumerate(
            text_chunks,
            start=1
        ):

            page_chunks.append({

                "chunk_id": chunk_id,
                "document": page["document"],
                "page": page_number,
                "section": section,
                "subsection": subsection,
                "chunk_number": chunk_number,
                "text": chunk_text

            })

            chunk_id += 1

    return page_chunks, current_subsection, chunk_id


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        documents = json.load(file)

    # Ensure numerical page ordering.
    documents.sort(
        key=lambda item: (
            item["document"],
            int(item["page"])
        )
    )

    all_chunks = []

    current_document = None
    current_subsection = None
    current_section = None
    chunk_id = 1

    for page in documents:

        document_name = page["document"]

        # Reset context for a new document.
        if document_name != current_document:

            current_document = document_name
            current_subsection = None
            current_section = None

        page_section = get_section(int(page["page"]))

        # Reset subsection context when the major section changes.
        if page_section != current_section:

            current_subsection = None
            current_section = page_section

        # Skip pages without usable extracted text.
        if len(page["text"].strip()) < 50:
            continue

        page_chunks, current_subsection, chunk_id = process_page(
            page,
            current_subsection,
            chunk_id
        )

        all_chunks.extend(page_chunks)

    # Save chunks and metadata.
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("=" * 55)
    print("CONTEXT-AWARE TEXT CHUNKING COMPLETED")
    print("=" * 55)

    print(f"Total extracted pages: {len(documents)}")
    print(f"Total usable chunks: {len(all_chunks)}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Overlap: {OVERLAP} characters")
    print("Metadata: document, page, section, subsection")
    print(f"Output: {OUTPUT_FILE}")

    print("\nSection distribution:")

    section_counts = {}

    for chunk in all_chunks:

        section = chunk["section"]

        section_counts[section] = (
            section_counts.get(section, 0) + 1
        )

    for section, count in section_counts.items():
        print(f"{section}: {count} chunks")


if __name__ == "__main__":
    main()
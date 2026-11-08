import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# ==================================================
# 1. CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Gemini API key not found.")

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.5-flash-lite"

REFUSAL = "I could not find this in the loaded documents."


# ==================================================
# 2. CITATION VALIDATION
# ==================================================

def format_citations(answer, passage_map):

    # Reject independently generated document citations.
    if re.search(
        r"\[Document:|\bPDF\s+PAGE\s+NUMBER:",
        answer,
        flags=re.IGNORECASE
    ):
        return REFUSAL

    citation_pattern = r"\[(P\d+(?:\s*,\s*P\d+)*)\]"

    matches = list(re.finditer(citation_pattern, answer))

    if not matches:
        return REFUSAL

    # Validate every passage ID.
    for match in matches:

        passage_ids = re.findall(r"P(\d+)", match.group(1))

        for passage_id in passage_ids:

            if int(passage_id) not in passage_map:
                return REFUSAL

    # Replace passage IDs with trusted document metadata.
    def replace_citation(match):

        passage_ids = re.findall(r"P(\d+)", match.group(1))

        citations = []

        for passage_id in passage_ids:

            passage = passage_map[int(passage_id)]

            citation = (
                f"[Document: {passage['document']}, "
                f"PDF page: {passage['page']}]"
            )

            if citation not in citations:
                citations.append(citation)

        return " ".join(citations)

    formatted_answer = re.sub(
        citation_pattern,
        replace_citation,
        answer
    )

    # Reject unresolved passage references.
    if re.search(r"\bP\d+\b", formatted_answer):
        return REFUSAL

    return formatted_answer


# ==================================================
# 3. ANSWER GENERATION
# ==================================================

def generate_answer(question, retrieved_passages):

    if not retrieved_passages:
        return REFUSAL

    # --------------------------------------------------
    # REMOVE EXACT DUPLICATES
    # --------------------------------------------------

    unique_passages = []
    seen = set()

    for passage in retrieved_passages:

        identity = (
            passage["document"],
            passage["page"],
            passage["text"].strip()
        )

        if identity not in seen:
            seen.add(identity)
            unique_passages.append(passage)

    # Give anticoagulant questions a wider evidence window.
    if "anticoagulant" in question.lower():
        selected_passages = unique_passages[:10]
    else:
        selected_passages = unique_passages[:6]

    # --------------------------------------------------
    # ASSIGN PASSAGE IDs AND INCLUDE SOURCE METADATA
    # --------------------------------------------------

    passage_map = {}
    context_parts = []

    for index, passage in enumerate(selected_passages, start=1):

        passage_map[index] = passage

        section = passage.get("section", "Not specified")
        subsection = passage.get("subsection", "Not specified")

        context_parts.append(
            f"PASSAGE ID: P{index}\n"
            f"DOCUMENT: {passage['document']}\n"
            f"PDF PAGE: {passage['page']}\n"
            f"SECTION: {section}\n"
            f"SUBSECTION: {subsection}\n"
            f"CONTENT:\n{passage['text']}"
        )

    context = "\n\n" + (
        "\n" + "=" * 50 + "\n"
    ).join(context_parts)

    # --------------------------------------------------
    # PROMPT
    # --------------------------------------------------

    prompt = f"""
You are MEDIQUERY, a document-grounded assistant for
Nigerian public health reference materials.

Answer the question using the supplied passages.

GROUNDING RULES:

1. Use only information explicitly supported by the passages.
   Do not introduce external medical knowledge.

2. If the passages contain relevant evidence that answers
   the question, provide the answer. Do not refuse simply
   because the evidence is distributed across multiple passages.

3. If the passages do not contain sufficient evidence,
   respond exactly:
   {REFUSAL}

4. Cite factual statements using the passage IDs provided.
   Use only this format: [P1], [P2], or [P1] [P2].

5. Never invent passage IDs, document names or page numbers.
   Do not generate document citations yourself.

6. Distinguish between the Main Adult List, PHC lists,
   PHC Health Centres, PHC Clinics, Health Posts and PPMV
   listings using the section metadata.

7. If the question explicitly requests the adult NEML,
   use ONLY passages belonging to the Main Adult List.
   Do not include PHC, Health Post, Health Centre,
   PHC Clinic or PPMV entries.

8. If the question requests information across multiple
   sections, organize the answer under separate section
   headings. Never merge entries from different sections.

9. For therapeutic-category questions, use the actual
   subsection heading as the primary evidence of category
   membership. Do not classify an individual medicine
   based solely on its proximity to other medicines.
   If the subsection contains unrelated medicines,
   distinguish them rather than automatically listing
   everything under that heading.

10. Do not invent formulations, strengths, indications,
    dosages or treatment recommendations.

11. Consolidate duplicate entries only when they refer
    to the same medicine and formulation.

12. The Nigerian Essential Medicines List is a medicine
    listing, not a comprehensive treatment guideline.

13. Do not provide patient-specific medical advice.

ANSWER STYLE:

- Answer the actual question directly.
- Use concise bullet points for medicine lists.
- Include relevant formulations and strengths where available.
- Preserve distinctions between different sections.
- Cite each factual bullet or statement.
- Do not add unnecessary introductory explanations.

SUPPLIED PASSAGES:

{context}

QUESTION:

{question}

ANSWER:
"""

    # --------------------------------------------------
    # GEMINI REQUEST WITH RETRIES
    # --------------------------------------------------

    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "temperature": 0.1
                }
            )

            answer = (response.text or "").strip()

            if not answer:
                return REFUSAL

            # Preserve the exact refusal response.
            if REFUSAL.lower() in answer.lower():
                return REFUSAL

            return format_citations(
                answer,
                passage_map
            )

        except Exception as error:

            message = str(error).lower()

            retryable = any(
                term in message
                for term in [
                    "503",
                    "unavailable",
                    "429",
                    "resource_exhausted"
                ]
            )

            if not retryable or attempt == max_attempts - 1:
                raise

            wait_seconds = 2 ** attempt

            print(
                f"Gemini temporarily unavailable. "
                f"Retrying in {wait_seconds}s..."
            )

            time.sleep(wait_seconds)


# ==================================================
# 4. LOCAL TESTING
# ==================================================

if __name__ == "__main__":

    from src.retrieve import Retriever

    retriever = Retriever()

    questions = [
        "Which anticoagulants are listed in the adult NEML?"
]

    for question in questions:

        print("\n" + "=" * 70)
        print("QUESTION:", question)

        passages = retriever.search(
            question,
            top_k=10
        )

        print("Retrieved passages:", len(passages))

        answer = generate_answer(question, passages)

        print("\nANSWER:")
        print(answer)
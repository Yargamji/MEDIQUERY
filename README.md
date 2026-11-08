# MEDIQUERY

**A document-grounded assistant for Nigerian public health reference
materials**

MEDIQUERY is an early-stage retrieval-augmented generation (RAG)
prototype. It retrieves relevant passages from an indexed public-health
reference document and uses a language model to produce concise answers
with document and PDF-page citations.

> **Important:** MEDIQUERY is a research/learning prototype, not a
> clinically validated decision-support system. It must not be used for
> patient-specific diagnosis, prescribing, or treatment decisions.

## Current scope

The currently indexed corpus is the **Nigeria 2024 Adult Essential
Medicines List (NEML)**.

The ingestion workflow scanned three PDFs (1,267 pages in total), but
only 103 pages from the Adult NEML yielded usable text. Two other PDFs
were image/scanned documents and were not OCR-processed. The current
searchable index contains **369 chunks**.

The NEML is a medicines listing, not a comprehensive treatment
guideline. The prototype should not be represented as covering all
Nigerian clinical guidelines.

## Main capabilities

-   Page-level PDF text extraction with document and PDF-page metadata.
-   Heading-aware text chunking with overlap.
-   Open-source text embeddings using `BAAI/bge-small-en-v1.5`.
-   Hybrid retrieval combining dense similarity and BM25.
-   Gemini-based answer generation grounded in retrieved passages.
-   Citation mapping from controlled passage IDs to source document and
    PDF page.
-   Exact refusal response when the loaded material does not support an
    answer.
-   Streamlit user interface.

## Technology stack

-   Python 3.12
-   PyPDF
-   FastEmbed (`BAAI/bge-small-en-v1.5`)
-   NumPy
-   BM25 retrieval
-   Google Gemini API (`google-genai`)
-   Streamlit

## Project structure

``` text
rag-health-assistant/
├── data/
│   ├── raw/
│   └── processed/
├── index/
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── chunk.py
│   ├── embed.py
│   ├── retrieve.py
│   └── generate.py
├── eval/
│   ├── questions.json
│   ├── run_evaluation.py
│   └── results/
├── app.py
├── requirements.txt
├── runtime.txt
└── README.md
```

## Setup

1.  Clone the repository.
2.  Create and activate a Python 3.12 environment.
3.  Install dependencies.
4.  Add your Gemini API key to a local `.env` file.

Example:

``` bash
conda create -n rag-health-assistant python=3.12 -y
conda activate rag-health-assistant
pip install -r requirements.txt
```

Create `.env` in the project root:

``` text
GEMINI_API_KEY=your_api_key_here
```

Do not commit `.env` or expose API keys.

## Build the local index

Run the pipeline from the project root:

``` bash
python -m src.ingest
python -m src.chunk
python -m src.embed
```

Use the exact script entry points in the checked-in source if they
differ from these commands.

## Run the application

``` bash
streamlit run app.py
```

## Evaluation snapshot

A 30-question evaluation set was used:

-   15 single-passage answerable questions
-   5 multi-passage answerable questions
-   5 paraphrased answerable questions
-   5 unanswerable questions

Baseline results:

  Metric                        Result
  ---------------------- -------------
  Retrieval Hit@1          60% (15/25)
  Retrieval Hit@4          84% (21/25)
  Refusal accuracy          100% (5/5)
  Mean answer score             0.84/2
  Citation correctness      36% (9/25)

Answer and citation scores are provisional manual assessments, not
independently validated. Hit@K records whether at least one gold passage
was retrieved; it does not establish complete evidence coverage for
multi-passage questions.

## Known limitations

-   Only the 2024 Adult NEML is currently indexed.
-   Scanned/image-only PDFs were not OCR-processed.
-   PDF tables and source text artefacts can affect extraction quality.
-   Retrieval can miss relevant adjacent passages.
-   The generator can omit evidence, mix sections, or misclassify
    medicines.
-   Citation correctness and answer quality require further independent
    evaluation.
-   No prospective clinical validation has been performed.

## Responsible use

MEDIQUERY is intended for demonstration and research into
document-grounded retrieval. Always consult the original source document
and current authoritative clinical guidance. Do not use MEDIQUERY as a
substitute for clinical judgement or as the basis for patient-specific
care.

## Evaluation files

Keep the question set, gold passage annotations, raw evaluation output,
and manually reviewed scores together in the `eval/` directory or a
clearly labelled results archive. Preserve the distinction between
automated retrieval metrics and provisional manual answer/citation
scoring.

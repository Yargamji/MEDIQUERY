import json
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------
# PATHS AND MODEL CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_PATH = PROJECT_ROOT / "index" / "embeddings.npy"
METADATA_PATH = PROJECT_ROOT / "index" / "metadata.json"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------
# TEXT PROCESSING CONFIGURATION
# ---------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "what", "which", "who", "whom", "where", "when",
    "how", "why", "does", "do", "did", "can", "could",
    "should", "would", "will", "may", "might",
    "of", "for", "to", "in", "on", "at", "by", "with",
    "from", "about", "and", "or", "but", "as", "if",
    "it", "its", "this", "that", "these", "those",
    "me", "my", "you", "your", "we", "our",
    "be", "been", "being", "have", "has", "had",
    "i", "there", "their", "they", "them",
    "nigerian", "nigeria", "essential", "neml"
}


PHRASE_REWRITES = {
    "high blood pressure": [
        "hypertension",
        "antihypertensive"
    ],

    "high bp": [
        "hypertension",
        "antihypertensive"
    ],

    "intestinal worms": [
        "intestinal helminths",
        "anthelmintics",
        "albendazole",
        "mebendazole"
    ],

    "worm infection": [
        "intestinal helminths",
        "anthelmintics",
        "albendazole",
        "mebendazole"
    ],

    "oral rehydration salts": [
        "oral rehydration solution",
        "ORS",
        "low osmolarity"
    ],

    "rehydration solution": [
        "oral rehydration salts",
        "ORS",
        "low osmolarity"
    ],

    "short acting bronchodilator": [
        "salbutamol",
        "bronchodilator"
    ]
}

SYNONYMS = {
    "hypertension": [
        "antihypertensive"
    ],

    "antihypertensive": [
        "hypertension"
    ],

    "worms": [
        "helminths",
        "anthelmintics"
    ],

    "helminths": [
        "worms",
        "anthelmintics"
    ],

    "ors": [
        "oral rehydration salts",
        "oral rehydration solution"
    ],

    "osmolarity": [
        "osmolality",
        "low osmolarity"
    ],

    "amoxicillin": [
        "antibiotic",
        "antibacterial"
    ]
}


GENERIC_TERMS = {
    "medicine", "medicines",
    "drug", "drugs",
    "treatment", "management",
    "therapy", "medication", "medications",
    "used", "use", "listed", "list",
    "available", "recommended", "recommendation",
    "indication", "indications",
    "patient", "patients",
    "condition", "conditions",
    "disease", "diseases",
    "infection", "infections",
    "formulation", "formulations",
    "strength", "strengths",
    "surgery", "surgical",
    "surgeon", "surgeons",
    "operation", "operations",
    "procedure", "procedures"
}


FRONT_MATTER_MARKERS = [
    "foreword",
    "acknowledgement",
    "acknowledgment",
    "preface",
    "message from",
    "table of contents"
]


# ---------------------------------------------------------
# TOKENIZATION AND QUERY EXPANSION
# ---------------------------------------------------------

def tokenize(text):
    """Convert text into lowercase tokens and remove stopwords."""

    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)

    return [
        token for token in tokens
        if token not in STOPWORDS and len(token) > 1
    ]


def rewrite_query(query):
    """
    Replace common clinical expressions with terminology
    more likely to appear in the source document.
    """

    rewritten = query.lower()

    for phrase, replacements in PHRASE_REWRITES.items():

        if phrase in rewritten:
            rewritten = rewritten.replace(
                phrase,
                " ".join(replacements)
            )

    return rewritten

def expand_tokens(tokens):
    """Add relevant synonyms to the query token list."""

    expanded = list(tokens)

    for token in tokens:

        if token in SYNONYMS:
            expanded.extend(
                tokenize(" ".join(SYNONYMS[token]))
            )

    return list(dict.fromkeys(expanded))


def contains_marker(text, markers):
    """Check whether a passage contains specified markers."""

    text = text.lower()

    return any(marker in text for marker in markers)


# ---------------------------------------------------------
# RETRIEVER
# ---------------------------------------------------------

class Retriever:

    def __init__(self):

        print("Loading retrieval resources...")

        # Load embeddings.
        self.embeddings = np.load(EMBEDDINGS_PATH)

        # Load metadata.
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        # Validate index consistency.
        if self.embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if len(self.metadata) != self.embeddings.shape[0]:
            raise ValueError(
                "Embedding and metadata counts do not match. "
                "Regenerate the embeddings."
            )

        if not all(
            "section" in item and "subsection" in item
            for item in self.metadata
        ):
            raise ValueError(
                "Section metadata is missing. "
                "Regenerate the index using src.embed."
            )

        # Load embedding model.
        self.embedding_model = TextEmbedding(
            model_name=EMBEDDING_MODEL
        )

        # Normalize embeddings for cosine similarity.
        norms = np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True
        )

        self.embeddings = self.embeddings / np.maximum(
            norms, 1e-10
        )

        # Build BM25 index.
        self.tokenized_corpus = [
            tokenize(item["text"])
            for item in self.metadata
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Build corpus vocabulary.
        self.corpus_vocabulary = set()

        for tokens in self.tokenized_corpus:
            self.corpus_vocabulary.update(tokens)

        print(
            f"Retriever ready: {len(self.metadata)} chunks loaded."
        )

    # -----------------------------------------------------
    # SEARCH
    # -----------------------------------------------------

    def search(self, query, top_k=5):

        rewritten_query = rewrite_query(query)

        query_tokens = tokenize(rewritten_query)

        if not query_tokens:
            return []

        # Expand query terminology.
        expanded_tokens = expand_tokens(query_tokens)

        # Identify meaningful query terms.
        anchor_tokens = [
            token for token in expanded_tokens
            if token not in GENERIC_TERMS
        ]

        # Retain anchors represented in the corpus.
        matched_anchors = [
            token for token in anchor_tokens
            if token in self.corpus_vocabulary
        ]

        # Avoid returning unrelated passages when no
        # meaningful query terms match the corpus.
        if not matched_anchors:
            return []

        matched_terms = [
            token for token in expanded_tokens
            if token in self.corpus_vocabulary
        ]

        if not matched_terms:
            return []

        # -------------------------------------------------
        # DENSE RETRIEVAL
        # -------------------------------------------------

        query_embedding = np.array(
            list(
                self.embedding_model.query_embed(
                    [rewritten_query]
                )
            )[0]
        )

        query_embedding = query_embedding / max(
            np.linalg.norm(query_embedding), 1e-10
        )

        dense_scores = np.dot(
            self.embeddings,
            query_embedding
        )

        dense_candidates = np.argsort(
            dense_scores
        )[::-1][:60]

        # -------------------------------------------------
        # BM25 RETRIEVAL
        # -------------------------------------------------

        bm25_scores = self.bm25.get_scores(expanded_tokens)

        bm25_candidates = np.argsort(
            bm25_scores
        )[::-1][:60]

        # -------------------------------------------------
        # RECIPROCAL RANK FUSION
        # -------------------------------------------------

        rrf_scores = {}

        # Dense retrieval contribution.
        for rank, idx in enumerate(dense_candidates):

            rrf_scores[int(idx)] = (
                rrf_scores.get(int(idx), 0)
                + 1.0 / (60 + rank + 1)
            )

        # BM25 receives a higher weight.
        for rank, idx in enumerate(bm25_candidates):

            rrf_scores[int(idx)] = (
                rrf_scores.get(int(idx), 0)
                + 2.0 / (60 + rank + 1)
            )

        # -------------------------------------------------
        # RERANKING
        # -------------------------------------------------

        results = []

        for idx, base_score in rrf_scores.items():

            item = self.metadata[idx]

            passage = item["text"]
            passage_lower = passage.lower()

            # Exclude index pages (PDF pages 97 onwards).
            if (
                item.get("document")
                == "nigeria-2024-final-neml-adult-8th-edition.pdf"
                and item.get("page", 0) >= 97
            ):
                continue

            # Exclude index passages from clinical retrieval.
            if re.search(r"\bindex\b", passage_lower):
                continue

            # Exclude passages without meaningful query overlap.
            passage_tokens = set(tokenize(passage))

            anchor_matches = [
                token for token in matched_anchors
                if token in passage_tokens
            ]

            if not anchor_matches:
                continue

            # -------------------------------------------------
            # LEXICAL AND ANCHOR COVERAGE
            # -------------------------------------------------

            anchor_coverage = (
                len(anchor_matches) / len(matched_anchors)
            )

            lexical_matches = [
                token for token in matched_terms
                if token in passage_tokens
            ]

            lexical_coverage = (
                len(lexical_matches) / len(matched_terms)
                if matched_terms else 0
            )

            score = base_score

            score += anchor_coverage * 0.025
            score += lexical_coverage * 0.015

            # -------------------------------------------------
            # NEW: SUBSECTION-AWARE RANKING
            # -------------------------------------------------

            subsection = item.get("subsection") or ""

            subsection_tokens = set(
                tokenize(subsection)
            )

            subsection_matches = [
                token for token in matched_anchors
                if token in subsection_tokens
            ]

            if subsection_matches:

                subsection_coverage = (
                    len(subsection_matches)
                    / len(matched_anchors)
                )

                score += subsection_coverage * 0.025

            # -------------------------------------------------
            # NEW: SECTION-AWARE RANKING
            # -------------------------------------------------

            section = item.get("section") or ""

            section_tokens = set(
                tokenize(section)
            )

            section_matches = [
                token for token in matched_anchors
                if token in section_tokens
            ]

            if section_matches:

                section_coverage = (
                    len(section_matches)
                    / len(matched_anchors)
                )

                score += section_coverage * 0.015

            # -------------------------------------------------
            # HEADING BOOST
            # -------------------------------------------------

            heading_patterns = [
                "amoxicillin",
                "antihypertensive",
                "anthelmintic",
                "albendazole",
                "mebendazole",
                "oral rehydration",
                "salbutamol"
            ]

            for heading in heading_patterns:

                if (
                    heading in passage_lower
                    and heading in rewritten_query
                ):
                    score += 0.02

            # -------------------------------------------------
            # FRONT MATTER PENALTY
            # -------------------------------------------------

            if contains_marker(
                passage_lower,
                FRONT_MATTER_MARKERS
            ):

                score *= 0.5

            # -------------------------------------------------
            # OSMOLARITY VALUE BOOST
            # -------------------------------------------------

            value_boost = 0.0

            if "osmolarity" in rewritten_query:

                # Strongly prioritize explicit numerical values.
                if re.search(
                    r"osmolarity\s*:\s*[\d.,]+\s*m",
                    passage_lower
                ):

                    value_boost = 0.05

                elif "osmolarity" in passage_lower:

                    value_boost = 0.005

            score += value_boost

            # -------------------------------------------------
            # GENERAL NUMERICAL VALUE BOOST
            # -------------------------------------------------

            value_terms = [
                "concentration",
                "dosage",
                "strength",
                "amount",
                "value"
            ]

            if any(
                term in rewritten_query
                for term in value_terms
            ):

                if re.search(r"\d", passage_lower):
                    score += 0.005

            # -------------------------------------------------
            # STORE RERANKED RESULT
            # -------------------------------------------------

            results.append({
                **item,
                "score": float(score)
            })

        # -------------------------------------------------
        # SORT RESULTS
        # -------------------------------------------------

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        # Limit duplicate chunks from the same PDF page.
        final_results = []

        page_counts = {}

        for result in results:

            page_key = (
                result.get("document"),
                result.get("page")
            )

            if page_counts.get(page_key, 0) >= 2:
                continue

            final_results.append(result)

            page_counts[page_key] = (
                page_counts.get(page_key, 0) + 1
            )

            if len(final_results) >= top_k:
                break

        return final_results


# ---------------------------------------------------------
# TEST RETRIEVAL
# ---------------------------------------------------------

if __name__ == "__main__":

    retriever = Retriever()

    test_queries = [

        "Which medicines are used for treating hypertension?",

        "Which medicines are used for high blood pressure?",

        "Which strengths of amoxicillin are listed?",

        "What is the recommended surgery for appendicitis?",

        "What is the osmolarity of the low-osmolarity ORS?",

        "Which medicines are listed for intestinal worm infections?"

    ]

    for query in test_queries:

        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        results = retriever.search(query, top_k=6)

        if not results:

            print("No relevant passages retrieved.")
            continue

        for rank, result in enumerate(results, start=1):

            print(f"\nRank: {rank}")
            print(f"Document: {result.get('document')}")
            print(f"PDF page: {result.get('page')}")
            print(f"Section: {result.get('section')}")
            print(f"Subsection: {result.get('subsection')}")
            print(f"Score: {result.get('score'):.5f}")
            print(f"Passage: {result.get('text')[:500]}...")
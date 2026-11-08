import json
import numpy as np
from pathlib import Path
from fastembed import TextEmbedding


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"
INDEX_DIR = PROJECT_ROOT / "index"

EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
METADATA_FILE = INDEX_DIR / "metadata.json"


# --------------------------------------------------
# EMBEDDING CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32


def main():

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. LOAD TEXT CHUNKS
    # --------------------------------------------------

    print("Loading text chunks...")

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("No text chunks found. Run the chunking module first.")

    texts = [chunk["text"] for chunk in chunks]

    print(f"Loaded {len(texts)} text chunks.")

    # --------------------------------------------------
    # 2. LOAD EMBEDDING MODEL
    # --------------------------------------------------

    print(f"\nLoading embedding model: {MODEL_NAME}")

    model = TextEmbedding(model_name=MODEL_NAME)

    # --------------------------------------------------
    # 3. GENERATE EMBEDDINGS
    # --------------------------------------------------

    print("\nGenerating embeddings...")

    embeddings = np.array(
        list(model.embed(texts, batch_size=BATCH_SIZE)),
        dtype=np.float32
    )

    # Validate embedding dimensions and alignment.
    if embeddings.ndim != 2:
        raise ValueError("Embedding output must be a 2D array.")

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            "Embedding and chunk counts do not match. "
            "Check the embedding generation process."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numerical values.")

    # --------------------------------------------------
    # 4. PRESERVE CHUNK METADATA
    # --------------------------------------------------

    metadata = [

        {
            "chunk_id": chunk["chunk_id"],
            "document": chunk["document"],
            "page": chunk["page"],
            "section": chunk["section"],
            "subsection": chunk["subsection"],
            "chunk_number": chunk["chunk_number"],
            "text": chunk["text"]
        }

        for chunk in chunks
    ]

    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            "Metadata and embedding counts do not match."
        )

    # --------------------------------------------------
    # 5. SAVE EMBEDDINGS AND METADATA
    # --------------------------------------------------

    np.save(EMBEDDINGS_FILE, embeddings)

    with open(METADATA_FILE, "w", encoding="utf-8") as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------
    # 6. DISPLAY RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("EMBEDDING GENERATION COMPLETED")
    print("=" * 55)

    print(f"Number of chunks: {len(chunks)}")
    print(f"Number of embeddings: {embeddings.shape[0]}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Embedding model: {MODEL_NAME}")

    print(f"\nEmbeddings saved to: {EMBEDDINGS_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")

    print("\nMetadata fields preserved:")
    print(", ".join(metadata[0].keys()))


if __name__ == "__main__":
    main()
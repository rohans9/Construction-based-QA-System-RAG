import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np

from config import get_openai_client, EMBEDDING_MODEL


def load_chunked_docs(json_path: str = "docs/chunks.json") -> List[Dict]:
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Chunked JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        docs = json.load(f)

    if not isinstance(docs, list):
        raise ValueError(f"Expected a list of documents in {path}, got {type(docs)}")

    return docs


def embed_texts(texts: List[str]) -> np.ndarray:
    client = get_openai_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


def build_faiss_index(docs: List[Dict]):
    """Build a FAISS index using OpenAI embeddings."""
    texts = [d["content"] for d in docs]
    embeddings = embed_texts(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings


def save_faiss_index(index, path: str = "faiss_index.bin") -> None:
    faiss.write_index(index, path)


if __name__ == "__main__":
    docs = load_chunked_docs("docs/chunks.json")
    print(f"Loaded {len(docs)} chunked docs for indexing with OpenAI embeddings.")

    index, embeddings = build_faiss_index(docs)
    save_faiss_index(index, "faiss_index.bin")

    print("FAISS index created with", len(docs), "entries")



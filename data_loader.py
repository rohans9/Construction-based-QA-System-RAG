import os
import json
from pathlib import Path
from typing import List, Dict
import PyPDF2


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def load_pdfs_from_folder(folder_path: str) -> List[Dict]:
    docs = []
    folder = Path(folder_path)

    for pdf_file in folder.glob("*.pdf"):
        content = extract_text_from_pdf(str(pdf_file))

        docs.append({
            "content": content,
            "metadata": {
                "file": pdf_file.name
            }
        })

    return docs


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def prepare_documents(pdf_folder_path: str) -> List[Dict]:
    raw_docs = load_pdfs_from_folder(pdf_folder_path)
    documents = []

    for doc in raw_docs:
        chunks = chunk_text(doc["content"])
        for c in chunks:
            documents.append({
                "content": c,
                "metadata": doc["metadata"]
            })

    return documents


def save_documents_to_json(documents: List[Dict], output_path: str) -> None:
    """Save to a JSON file."""
    path = Path(output_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    pdf_folder = "docs"
    docs = prepare_documents(pdf_folder)
    print("Total chunked documents:", len(docs))

    output_json = Path("docs") / "chunks.json"
    save_documents_to_json(docs, str(output_json))
    print(f"Saved JSON dataset to: {output_json}")

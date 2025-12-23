# Construction based QA System (RAG)

Lightweight Retrieval-Augmented Generation (RAG) demo that:
- Chunks local PDFs
- Builds OpenAI embeddings + FAISS index
- Answers questions strictly from retrieved context

## How it works
- `data_loader.py` extracts text from `docs/*.pdf`, chunks it, and saves `docs/chunks.json`.
- `embeddings.py` embeds each chunk with OpenAI and builds a FAISS index (`faiss_index.bin`).
- `rag_engine.py` loads the chunks + index, retrieves top matches for a query, and asks the chat model to respond only with that context.

## Setup
1) Python 3.10+ recommended.
2) Install deps:
   - `pip install -r requirements.txt`
3) Set your API key (e.g. in a `.env` file):
   - `OPENAI_API_KEY=sk-...`

## Prepare data
1) Drop PDFs into `docs/` (sample `doc1.pdf` included).
2) Generate chunks JSON:
   - `python data_loader.py`
   - Outputs `docs/chunks.json`

## Build the index
- `python embeddings.py`
- Produces `faiss_index.bin` (FAISS) used at query time.

## Ask questions
- `python rag_engine.py`
- Edit the `questions` list at the bottom of `rag_engine.py` or import `RAGPipeline` in your own script:
```python
from rag_engine import RAGPipeline
rag = RAGPipeline()
print(rag.answer("What factors affect construction project delays?", k=3))
```

## Config knobs
- Models are set in `config.py` (`EMBEDDING_MODEL`, `CHAT_MODEL`).
- Retrieval depth via `k` in `rag_engine.py`.

## Notes
- Retrieval context can be printed by uncommenting the print line in `RAGPipeline.answer`.
- Ensure the PDFs are primarily text (scanned images need OCR first).
- If you change docs, rerun `data_loader.py` then `embeddings.py` to refresh the index.

---

## Open-source models notebook
File: `RAG using open-source models.ipynb` — a Colab-friendly version that avoids OpenAI APIs and stays under ~1B parameter models.

What it does:
- Installs `faiss-gpu`, `PyPDF2`, and uses Hugging Face models.
- Embeddings: `BAAI/bge-small-en-v1.5` via `sentence_transformers`.
- LLM: `Qwen/Qwen1.5-0.5B-Chat` loaded with token auth (`HF_TOKEN`) and `device_map="auto"`.
- Chunks PDFs from `/content/documents`, saves `chunks.json`, builds a cosine FAISS index, and runs a small RAG loop.

Quick run (Colab-style):
1) Set secret `HF_TOKEN` in Colab (for gated models if needed).
2) Upload PDFs to `/content/documents`.
3) Run cells to generate `chunks.json`, build `faiss.index`, and load the Qwen chat model.
4) Ask questions with `rag.answer("your question", k=3)`. The prompt enforces context-only answers and to return "Information not found" when unsupported.

Tips:
- If you change docs, re-run chunking + indexing cells before querying.
- GPU is assumed for the notebook; drop to CPU by removing `device_map="auto"` (will be slower).
- If you have larger/faster GPUs, swap in bigger open-source models (higher parameter counts) for better quality. I stayed with sub-1B models due to limited GPU.

### Observed behavior
- The small open-source stack (bge-small + Qwen 0.5B) tended to hallucinate and often missed answers even with context.
- The OpenAI frontier setup answered well when context existed and politely said it couldn’t help when context was insufficient.
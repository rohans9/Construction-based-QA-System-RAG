import faiss
import numpy as np
from typing import List

from config import get_openai_client, EMBEDDING_MODEL, CHAT_MODEL
from embeddings import load_chunked_docs


class RAGPipeline:
    """
    This RAG pipeline uses:
      - OpenAI embeddings for retrieval
      - OpenAI chat model for answer generation

    It assumes:
      - docs/chunks.json exists (from data_loader.py)
      - openai_faiss_index.bin exists (from openai_build_index.py)
    """

    def __init__(
        self,
        index_path: str = "faiss_index.bin",
        docs_path: str = "docs/chunks.json",
    ):
        self.client = get_openai_client()
        self.docs = load_chunked_docs(docs_path)
        self.index = faiss.read_index(index_path)

    def _embed_query(self, query: str) -> np.ndarray:
        """Embedding the user query using the same OpenAI embedding model."""
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query],
        )
        vec = np.array([response.data[0].embedding], dtype="float32")
        return vec

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve top-k most similar chunks to the query."""
        query_vec = self._embed_query(query)
        distances, indices = self.index.search(query_vec, k)
        return [self.docs[i]["content"] for i in indices[0]]

    def answer(self, question: str, k: int = 3) -> str:
        """Generate an answer using retrieved context and OpenAI chat model."""
        chunks = self.retrieve(question, k=k)
        context = "\n\n".join(chunks)

        """If in case to print retrieved context"""
        # print("Retrieved context:\n", context, "\n") 

        system_prompt = (
            """You are a helpful assistant that answers questions based ONLY on the provided context.
            If the answer is not clearly contained in the context, respond politely that you cannot
            help with that specific point"""
        )

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer concisely based only on the context above."
        )

        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )

        return response.choices[0].message.content


if __name__ == "__main__":
    rag = RAGPipeline()
    questions = ["What factors affect construction project delays?","Explain the pricing plans"]
    for question in questions:
        print("Q:", question)
        print("A:", rag.answer(question))
        print()
        



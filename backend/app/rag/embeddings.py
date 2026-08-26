from typing import List
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from app.core.config import settings
from app.core.logging import logger


class AnalyticaEmbeddings:
    """
    LangChain-native embeddings using HuggingFaceEndpointEmbeddings.
    Calls the HF Inference API feature-extraction endpoint for real semantic vectors.
    Falls back to lightweight hash-based vectors when API is unavailable,
    ensuring zero heavy model loads on local hardware.
    """

    def __init__(self):
        self.api_token = settings.HUGGINGFACEHUB_API_TOKEN
        self.model_name = settings.HF_EMBEDDING_MODEL_ID or "sentence-transformers/all-MiniLM-L6-v2"
        self._lc_embeddings: HuggingFaceEndpointEmbeddings | None = None

        if self.api_token and len(self.api_token) > 5:
            try:
                self._lc_embeddings = HuggingFaceEndpointEmbeddings(
                    model=self.model_name,
                    huggingfacehub_api_token=self.api_token,
                )
                logger.info(f"Initialized HuggingFaceEndpointEmbeddings with model '{self.model_name}'")
            except Exception as e:
                logger.warning(f"Failed to initialize HuggingFaceEndpointEmbeddings: {e}. Using fallback embeddings.")
        else:
            logger.warning("HUGGINGFACEHUB_API_TOKEN not set — using fallback hash-based embeddings.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._lc_embeddings:
            try:
                return self._lc_embeddings.embed_documents(texts)
            except Exception as e:
                logger.warning(f"HuggingFaceEndpointEmbeddings.embed_documents failed: {e}. Using fallback.")
        return [self._fallback_embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        if self._lc_embeddings:
            try:
                return self._lc_embeddings.embed_query(text)
            except Exception as e:
                logger.warning(f"HuggingFaceEndpointEmbeddings.embed_query failed: {e}. Using fallback.")
        return self._fallback_embed(text)

    def _fallback_embed(self, text: str, dim: int = 384) -> List[float]:
        """Lightweight token-frequency vector — no external calls, no local model."""
        vec = [0.0] * dim
        words = text.lower().split()
        for idx, word in enumerate(words):
            hash_val = hash(word) % dim
            vec[hash_val] += 1.0 / (idx + 1)
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


# Module-level alias kept for backward compatibility with vectorstore.py
get_embeddings = AnalyticaEmbeddings

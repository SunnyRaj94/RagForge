import os
import httpx
from typing import List
from src.ragforge.utils.logging_hooks import observe_stage

from src.ragforge.config import OLLAMA_URL, DEFAULT_EMBEDDING_MODEL

default_model = DEFAULT_EMBEDDING_MODEL
ollama_url = OLLAMA_URL


@observe_stage("get_text_embedding")
def get_embedding(text: str, model_id: str = default_model) -> List[float]:
    """
    Generates an embedding vector using Ollama's embeddings API.
    """
    try:
        response = httpx.post(
            f"{ollama_url}/api/embeddings",
            json={"model": model_id, "prompt": text},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        raise RuntimeError(
            f"Failed to generate embedding via Ollama model '{model_id}': {str(e)}"
        )

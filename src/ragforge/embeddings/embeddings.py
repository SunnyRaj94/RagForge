import httpx
from typing import List
from ragforge.utils.logging_hooks import observe_stage
from ragforge.logging import get_logger

from ragforge.config import OLLAMA_URL, DEFAULT_EMBEDDING_MODEL

default_model = DEFAULT_EMBEDDING_MODEL
ollama_url = OLLAMA_URL
logger = get_logger(__name__)


@observe_stage("get_text_embedding")
def get_embedding(text: str, model_id: str = default_model) -> List[float]:
    """
    Generates an embedding vector using Ollama's embeddings API.
    """
    try:
        # Enforce safe length truncation to prevent Ollama token limit / physical batch size errors
        safe_text = text or ""
        if len(safe_text) > 5000:
            logger.warning(
                f"Embedding text length ({len(safe_text)} chars) exceeds safe threshold (5000). "
                f"Truncating to avoid Ollama token limit issues."
            )
            safe_text = safe_text[:5000]

        response = httpx.post(
            f"{ollama_url}/api/embeddings",
            json={"model": model_id, "prompt": safe_text},
            timeout=30.0,
        )
        response.raise_for_status()
        logger.info({"event": "get_embedding_succeeded", "model": model_id})
        return response.json()["embedding"]
    except Exception as e:
        logger.error({"event": "get_embedding_failed", "error": str(e)})
        raise RuntimeError(
            f"Failed to generate embedding via Ollama model '{model_id}': {str(e)}"
        )

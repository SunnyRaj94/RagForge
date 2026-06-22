from mcp.server.fastmcp import FastMCP
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

# Create FastMCP server
mcp = FastMCP("qdrant-server")

from src.ragforge.config import (
    QDRANT_URL,
    OLLAMA_URL,
    DEFAULT_EMBEDDING_MODEL,
    CHAT_HISTORY_COLLECTION,
    DEFAULT_COLLECTION,
)

# Connect to Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL)

# Connect to Ollama
ollama_url = OLLAMA_URL


def get_embeddings(text: str) -> list[float]:
    """Generates embedding vector from local Ollama model."""
    import os

    model_id = os.getenv("DEFAULT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    response = httpx.post(
        f"{ollama_url}/api/embeddings",
        json={"model": model_id, "prompt": text},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["embedding"]


@mcp.tool()
def create_collection(
    collection_name: str = "ragforge-collection", vector_size: int = 768
) -> str:
    """Create a vector collection in Qdrant with specified name and vector dimension size."""
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return f"Collection '{collection_name}' created successfully with dimension {vector_size}."
    except Exception as e:
        return f"Error creating collection: {str(e)}"


@mcp.tool()
def upsert_documents(
    collection_name: str = DEFAULT_COLLECTION,
    documents: list[dict] = None,
    session_id: str | None = None,
) -> str:
    """
    Upsert list of documents into Qdrant.
    Each document format: {"id": "optional-id", "text": "document text", "metadata": {}}

    IMPORTANT: Do NOT specify or change collection_name. Always use the default value.
    Use session_id parameter to tag documents for the current session.
    """

    try:
        points = []
        for idx, doc in enumerate(documents):
            # Parse or auto-generate point ID
            raw_id = doc.get("id")
            if not raw_id:
                doc_id = str(uuid.uuid4())
            elif isinstance(raw_id, int):
                doc_id = raw_id
            else:
                try:
                    uuid.UUID(str(raw_id))
                    doc_id = str(raw_id)
                except ValueError:
                    # Deterministically hash arbitrary string to a valid UUID
                    doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)))

            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            if session_id:
                metadata["session_id"] = session_id

            # Generate embedding
            vector = get_embeddings(text)

            # Form payload matching original text
            payload = {"text": text, **metadata}
            points.append(PointStruct(id=doc_id, vector=vector, payload=payload))

        # Ensure collection exists
        if not qdrant_client.collection_exists(collection_name):
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

        qdrant_client.upsert(collection_name=collection_name, points=points)
        return f"Successfully upserted {len(documents)} documents into collection '{collection_name}'."
    except Exception as e:
        return f"Error upserting documents: {str(e)}"


@mcp.tool()
def search_documents(
    collection_name: str = DEFAULT_COLLECTION,
    query: str = "",
    session_id: str | None = None,
    limit: int = 5,
) -> str:
    """
    Query vector database using a text prompt.
    Returns matched documents with metadata attributes and score metric.

    IMPORTANT: Do NOT specify or change collection_name. Always use the default value.
    Use session_id parameter to filter results for the current session.
    """

    try:
        if not qdrant_client.collection_exists(collection_name):
            return (
                f"No documents found (collection '{collection_name}' does not exist)."
            )

        from qdrant_client.models import (
            Filter,
            FieldCondition,
            MatchValue,
            IsEmptyCondition,
            PayloadField,
        )

        query_filter = None
        if session_id:
            query_filter = Filter(
                should=[
                    FieldCondition(
                        key="session_id", match=MatchValue(value=session_id)
                    ),
                    IsEmptyCondition(is_empty=PayloadField(key="session_id")),
                ]
            )

        query_vector = get_embeddings(query)
        response = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )
        hits = response.points

        results = []
        for hit in hits:
            payload = hit.payload or {}
            text = payload.get("text", "")
            # Filter text out of the metadata dictionary
            metadata = {k: v for k, v in payload.items() if k != "text"}
            results.append(f"- [Score: {hit.score:.4f}] {text}\n  Lineage: {metadata}")

        if not results:
            return "No matching documents found."

        return "\n\n".join(results)
    except Exception as e:
        return f"Error querying documents: {str(e)}"


@mcp.tool()
def search_chat_history(
    query: str = "", session_id: str | None = None, limit: int = 5
) -> str:
    """
    Search past chat history and conversations for context using semantic search.
    Use this to recall details from previous chats or older messages in the current session.
    Optional session_id filters results to only matching sessions.
    """
    try:
        # Create chat history collection if not exists (768 dimensions for nomic-embed-text)
        if not qdrant_client.collection_exists(CHAT_HISTORY_COLLECTION):
            qdrant_client.create_collection(
                collection_name=CHAT_HISTORY_COLLECTION,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_filter = None
        if session_id:
            query_filter = Filter(
                must=[
                    FieldCondition(key="session_id", match=MatchValue(value=session_id))
                ]
            )

        query_vector = get_embeddings(query)
        response = qdrant_client.query_points(
            collection_name=CHAT_HISTORY_COLLECTION,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )
        hits = response.points

        results = []
        for hit in hits:
            payload = hit.payload or {}
            text = payload.get("text", "")
            # Filter text out of the metadata dictionary
            metadata = {k: v for k, v in payload.items() if k != "text"}
            results.append(f"- [Score: {hit.score:.4f}] {text}\n  Lineage: {metadata}")

        if not results:
            return "No matching chat history found."

        return "\n\n".join(results)
    except Exception as e:
        return f"Error querying chat history: {str(e)}"


if __name__ == "__main__":
    mcp.run()

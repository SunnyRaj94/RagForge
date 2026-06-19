from mcp.server.fastmcp import FastMCP
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

# Create FastMCP server
mcp = FastMCP("qdrant-server")

from src.ragforge.config import QDRANT_URL, OLLAMA_URL, DEFAULT_EMBEDDING_MODEL

# Connect to Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL)

# Connect to Ollama
ollama_url = OLLAMA_URL


def get_embeddings(text: str) -> list[float]:
    """Generates embedding vector from local Ollama model."""
    response = httpx.post(
        f"{ollama_url}/api/embeddings",
        json={"model": DEFAULT_EMBEDDING_MODEL, "prompt": text},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["embedding"]


@mcp.tool()
def create_collection(collection_name: str, vector_size: int = 768) -> str:
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
def upsert_documents(collection_name: str, documents: list[dict]) -> str:
    """
    Upsert list of documents into Qdrant.
    Each document format: {"id": "optional-id", "text": "document text", "metadata": {}}
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

            # Generate embedding
            vector = get_embeddings(text)

            # Form payload matching original text
            payload = {"text": text, **metadata}
            points.append(PointStruct(id=doc_id, vector=vector, payload=payload))

        qdrant_client.upsert(collection_name=collection_name, points=points)
        return f"Successfully upserted {len(documents)} documents into collection '{collection_name}'."
    except Exception as e:
        return f"Error upserting documents: {str(e)}"


@mcp.tool()
def search_documents(collection_name: str, query: str, limit: int = 5) -> str:
    """
    Query vector database using a text prompt.
    Returns matched documents with metadata attributes and score metric.
    """
    try:
        query_vector = get_embeddings(query)
        response = qdrant_client.query_points(
            collection_name=collection_name, query=query_vector, limit=limit
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


if __name__ == "__main__":
    mcp.run()

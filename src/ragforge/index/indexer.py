import os
import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.ragforge.embeddings.embeddings import get_embedding
from src.ragforge.utils.logging_hooks import observe_stage

from src.ragforge.config import QDRANT_URL

qdrant_url = QDRANT_URL


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=qdrant_url)


@observe_stage("create_collection")
def create_collection(collection_name: str, vector_size: int = 768) -> str:
    """Creates a collection in Qdrant if it does not already exist."""
    client = get_qdrant_client()
    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            return f"Collection '{collection_name}' created successfully."
        return f"Collection '{collection_name}' already exists."
    except Exception as e:
        raise RuntimeError(f"Error creating collection '{collection_name}': {str(e)}")


@observe_stage("upsert_documents")
def upsert_documents(collection_name: str, documents: List[Dict[str, Any]]) -> str:
    """
    Upserts a list of document chunks into a Qdrant collection.
    Generates embedding vectors dynamically.
    """
    client = get_qdrant_client()
    try:
        points = []
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            raw_id = doc.get("id")

            # Form ID
            if not raw_id:
                doc_id = str(uuid.uuid4())
            elif isinstance(raw_id, int):
                doc_id = raw_id
            else:
                try:
                    uuid.UUID(str(raw_id))
                    doc_id = str(raw_id)
                except ValueError:
                    doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)))

            # Generate embedding
            vector = get_embedding(text)

            # Create payload
            payload = {"text": text, **metadata}

            points.append(PointStruct(id=doc_id, vector=vector, payload=payload))

        client.upsert(collection_name=collection_name, points=points)
        return f"Successfully upserted {len(documents)} documents into '{collection_name}'."
    except Exception as e:
        raise RuntimeError(f"Error upserting documents: {str(e)}")

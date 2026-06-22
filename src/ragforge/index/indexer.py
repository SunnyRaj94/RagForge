from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from ragforge.ingestion.indexing.qdrant import (
    QdrantVectorStore,
    create_collection as _create_collection,
    get_qdrant_client,
    search_documents as _search_documents,
    upsert_documents as _upsert_documents,
)


def create_collection(collection_name: str, vector_size: int = 768) -> str:
    return _create_collection(collection_name, vector_size)


def upsert_documents(collection_name: str, documents: list[dict[str, Any]]) -> str:
    return _upsert_documents(collection_name, documents)


def search_documents(
    collection_name: str, query: str, session_id: str | None = None, limit: int = 5
) -> str:
    return _search_documents(collection_name, query, session_id=session_id, limit=limit)


__all__ = [
    "QdrantClient",
    "QdrantVectorStore",
    "get_qdrant_client",
    "create_collection",
    "upsert_documents",
    "search_documents",
]

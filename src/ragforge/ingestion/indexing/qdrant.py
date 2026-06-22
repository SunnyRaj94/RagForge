from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ragforge.config import QDRANT_URL
from ragforge.embeddings.embeddings import get_embedding
from ragforge.logging import get_logger
from ragforge.ingestion.utils import summarize_text

logger = get_logger("ragforge.ingestion.qdrant")


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def _vector_config(vector_size: int) -> dict[str, VectorParams]:
    return {
        "text_vector": VectorParams(size=vector_size, distance=Distance.COSINE),
        "summary_vector": VectorParams(size=vector_size, distance=Distance.COSINE),
    }


def _ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    try:
        from qdrant_client.models import PayloadSchemaType

        for field_name, schema_type in [
            ("doc_id", PayloadSchemaType.KEYWORD),
            ("corpus_id", PayloadSchemaType.KEYWORD),
            ("tenant_id", PayloadSchemaType.KEYWORD),
            ("source_kind", PayloadSchemaType.KEYWORD),
            ("file_type", PayloadSchemaType.KEYWORD),
            ("page", PayloadSchemaType.INTEGER),
            ("slide_number", PayloadSchemaType.INTEGER),
            ("chunk_index", PayloadSchemaType.INTEGER),
            ("session_id", PayloadSchemaType.KEYWORD),
        ]:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception:
                pass
    except Exception:
        pass


def create_collection(collection_name: str, vector_size: int = 768) -> str:
    client = get_qdrant_client()
    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=_vector_config(vector_size),
            )
            _ensure_payload_indexes(client, collection_name)
            return f"Collection '{collection_name}' created successfully."
        return f"Collection '{collection_name}' already exists."
    except Exception as exc:
        raise RuntimeError(
            f"Error creating collection '{collection_name}': {exc}"
        ) from exc


def _point_id(raw_id: Any) -> str:
    if not raw_id:
        return str(uuid.uuid4())
    if isinstance(raw_id, int):
        return str(raw_id)
    try:
        uuid.UUID(str(raw_id))
        return str(raw_id)
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)))


def upsert_documents(collection_name: str, documents: list[dict[str, Any]]) -> str:
    logger.info({"event": "upsert_documents_started", "collection": collection_name, "num_docs": len(documents)})
    client = get_qdrant_client()
    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=_vector_config(768),
            )
        logger.info({"event": "collection_created", "collection": collection_name})
        _ensure_payload_indexes(client, collection_name)
        logger.info({"event": "payload_indexes_ensured", "collection": collection_name})

        points: list[PointStruct] = []
        for doc in documents:
            logger.info({"event": "processing_document", "doc_id": doc.get("id")})
            metadata = dict(doc.get("metadata", {}) or {})
            text = (
                doc.get("text")
                or metadata.get("retrieval_text")
                or metadata.get("chunk_summary")
                or ""
            )
            logger.info({"event": "text_extracted", "doc_id": doc.get("id"), "text_length": len(text)})
            summary = metadata.get("chunk_summary") or summarize_text(text)
            raw_text = doc.get("raw_text") or metadata.get("raw_text") or text
            raw_id = doc.get("id") or metadata.get("chunk_id")
            point_id = _point_id(raw_id)
            text_vector = get_embedding(text)
            summary_vector = get_embedding(summary)
            payload = {
                "text": raw_text,
                "summary": summary,
                **metadata,
            }
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "text_vector": text_vector,
                        "summary_vector": summary_vector,
                    },
                    payload=payload,
                )
            )
            logger.info({"event": "point_prepared", "doc_id": doc.get("id"), "point_id": point_id})
        client.upsert(collection_name=collection_name, points=points)
        return (
            f"Successfully upserted {len(points)} documents into '{collection_name}'."
        )
    except Exception as exc:
        logger.info({"event": "upsert_documents_failed", "error": str(exc)})
        raise RuntimeError(f"Error upserting documents: {exc}") from exc


def search_documents(
    collection_name: str,
    query: str,
    session_id: str | None = None,
    limit: int = 5,
) -> str:
    client = get_qdrant_client()
    logger = get_logger("ragforge.ingestion.qdrant")
    try:
        if not client.collection_exists(collection_name):
            return (
                f"No documents found (collection '{collection_name}' does not exist)."
            )

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if session_id:
            query_filter = Filter(
                must=[
                    FieldCondition(key="session_id", match=MatchValue(value=session_id))
                ]
            )

        query_vector = get_embedding(query)
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="text_vector",
            query_filter=query_filter,
            limit=limit,
        )
        results = []
        for hit in response.points:
            payload = hit.payload or {}
            text = payload.get("text", "")
            metadata = {k: v for k, v in payload.items() if k != "text"}
            results.append(f"- [Score: {hit.score:.4f}] {text}\n  Lineage: {metadata}")
        if not results:
            return "No matching documents found."
        return "\n\n".join(results)
    except Exception as exc:
        logger.info({"event": "search_failed", "error": str(exc)})
        raise RuntimeError(f"Error querying documents: {exc}") from exc


class QdrantVectorStore:
    def create_collection(self, collection_name: str, vector_size: int = 768) -> str:
        return create_collection(collection_name, vector_size)

    def upsert_documents(
        self, collection_name: str, documents: list[dict[str, Any]]
    ) -> str:
        return upsert_documents(collection_name, documents)

    def search_documents(
        self, collection_name: str, query: str, limit: int = 5, **kwargs: Any
    ) -> str:
        return search_documents(collection_name, query, limit=limit, **kwargs)

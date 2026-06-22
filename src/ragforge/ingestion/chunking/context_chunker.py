from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ragforge.loader.loader import load_chunking_config
from ragforge.ingestion.chunking.table_chunker import split_table_markdown
from ragforge.ingestion.chunking.text_chunker import split_text_with_overlap
from ragforge.logging import get_logger
from ragforge.ingestion.utils import (
    build_retrieval_text,
    extract_important_terms,
    normalize_whitespace,
    summarize_text,
)


def _extract_block_payload(
    doc: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    metadata = doc.get("metadata", {}) or {}
    content = doc.get("content", {}) or {}
    text = content.get("text") or content.get("ocr_text") or doc.get("text") or ""
    return normalize_whitespace(text), metadata, content


def chunk_documents(
    documents: list[dict[Any, Any]], config_path: str | None = None
) -> list[dict[str, Any]]:
    logger = get_logger("ragforge.ingestion.chunking")
    config = load_chunking_config(config_path)
    default_config = config.get("default", {"chunk_size": 800, "chunk_overlap": 100})
    chunked_docs: list[dict[str, Any]] = []
    logger.info(f"Starting document chunking process, Total documents: {len(documents)}")

    for doc in documents:
        text, metadata, content = _extract_block_payload(doc)
        file_type = metadata.get("file_type", "default")
        loading_strategy = doc.get(
            "loading_strategy", metadata.get("loading_strategy", "text")
        )
        type_config = config.get(file_type, default_config)
        chunk_size = type_config.get(
            "chunk_size", default_config.get("chunk_size", 800)
        )
        chunk_overlap = type_config.get(
            "chunk_overlap", default_config.get("chunk_overlap", 100)
        )
        tables = content.get("tables", []) if isinstance(content, dict) else []

        if not text and tables:
            raw_chunks = [
                table.get("markdown", "") for table in tables if table.get("markdown")
            ]
            raw_chunks = [chunk for chunk in raw_chunks if chunk.strip()]
        elif tables and len(text) > chunk_size:
            raw_chunks = split_table_markdown(text, chunk_size)
        elif len(text) > chunk_size:
            raw_chunks = split_text_with_overlap(text, chunk_size, chunk_overlap)
        else:
            raw_chunks = [text] if text else []

        if not raw_chunks and text:
            raw_chunks = [text]

        for idx, raw_text in enumerate(raw_chunks):
            raw_text = normalize_whitespace(raw_text)
            if not raw_text:
                continue
            chunk_summary = summarize_text(raw_text)
            important_terms = extract_important_terms(raw_text)
            retrieval_text = build_retrieval_text(
                chunk_summary, important_terms, raw_text
            )
            new_meta = dict(metadata)
            new_meta.update(
                {
                    "chunk_index": idx,
                    "loading_strategy": loading_strategy,
                    "retrieval_text": retrieval_text,
                    "chunk_summary": chunk_summary,
                    "important_terms": important_terms,
                }
            )
            chunked_docs.append(
                {
                    "id": doc.get("id") or str(uuid4()),
                    "text": retrieval_text,
                    "raw_text": raw_text,
                    "retrieval_text": retrieval_text,
                    "metadata": new_meta,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            logger.info(
                {
                    "event": "chunk_created",
                    "doc_id": new_meta.get("doc_id") or doc.get("doc_id"),
                    "chunk_index": idx,
                    "file_type": file_type,
                }
            )

    return chunked_docs

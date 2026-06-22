from __future__ import annotations

from typing import Any

import yaml

from ragforge.ingestion.chunking.context_chunker import (
    chunk_documents as _chunk_documents,
)
from ragforge.loader.loader import load_chunking_config


def split_text_by_chars(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    from ragforge.ingestion.chunking.text_chunker import split_text_with_overlap

    return split_text_with_overlap(text, chunk_size, chunk_overlap)


def chunk_documents_legacy(
    documents: list[dict[Any, Any]], config_path: str | None = None
) -> list[dict[str, Any]]:
    return _chunk_documents(documents, config_path=config_path)


# Backward-compatible alias used by existing imports.
chunk_documents = chunk_documents_legacy

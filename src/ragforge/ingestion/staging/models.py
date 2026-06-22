from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class StagedSourceDocument(BaseModel):
    doc_id: str
    logical_key: str
    source_name: str
    source_uri: str
    source_kind: str
    content_hash: str
    version: str | int | None = None
    status: str = "active"
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp_added: datetime
    timestamp_modified: datetime


class StagedPageBlock(BaseModel):
    block_id: str
    doc_id: str
    page_num: int | None = None
    loading_strategy: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp_added: datetime


class StagedChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp_added: datetime

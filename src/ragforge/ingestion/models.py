from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from typing import Any

from pydantic import BaseModel, Field


class PageContent(BaseModel):
    tables: list[dict[str, Any]] = Field(default_factory=list)
    text: str = ""
    images: list[str] = Field(default_factory=list)
    ocr_text: str | None = None
    llm_summary: str | None = None
    important_terms: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)


class LoadedBlock(BaseModel):
    doc_id: str
    page_num: int | None = None
    loading_strategy: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: PageContent
    text: str = ""

    def to_legacy_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["text"] = self.text or self.content.text or self.content.ocr_text or ""
        return data


class ChunkRecord(BaseModel):
    doc_id: str
    chunk_id: str
    chunk_index: int
    chunk_text: str
    chunk_summary: str | None = None
    important_terms: list[str] = Field(default_factory=list)
    retrieval_text: str = ""
    text_vector: list[float] | None = None
    summary_vector: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class RoutingDecision(BaseModel):
    loading_strategy: str
    reason: str
    confidence: float = 0.0


@dataclass(slots=True)
class DocumentUnit:
    doc_id: str
    source_path: str
    source_name: str
    file_type: str
    unit_num: int
    raw_text: str = ""
    tables: list[dict[str, Any]] = field(default_factory=list)
    image_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    loading_strategy_hint: str | None = None
    ocr_image_bytes: bytes | None = None
    ocr_image_name: str | None = None
    unit_kind: str = "page"

    @property
    def display_name(self) -> str:
        return self.metadata.get("filename") or Path(self.source_path).name

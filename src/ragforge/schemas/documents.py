from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    doc_id: str
    source_name: str
    source_uri: str
    source_kind: str
    mime_type: str | None = None
    file_ext: str | None = None
    corpus_id: str
    tenant_id: str | None = None
    version: str | int | None = None
    content_hash: str
    timestamp_added: datetime
    timestamp_modified: datetime
    other_details: str = ""
    other_details_format: str = "kvpipe"
    schema_version: str = "v1"
    doc_summary: str | None = None
    language: str | None = None
    token_count: int | None = None
    char_count: int | None = None


class DocumentChunk(BaseModel):
    doc_id: str
    chunk_id: str
    chunk_index: int
    chunk_text: str
    chunk_summary: str | None = None
    text_vector: list[float] | None = None
    summary_vector: list[float] | None = None
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    source_name: str
    source_uri: str
    source_kind: str
    corpus_id: str
    tenant_id: str | None = None
    content_hash: str
    chunk_hash: str
    timestamp_added: datetime
    timestamp_modified: datetime
    ingested_at: datetime
    language: str | None = None
    token_count: int | None = None
    char_count: int | None = None
    embedding_model: str | None = None
    embedding_version: str | None = None
    embedding_dim: int | None = None
    page_number: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    sheet_name: str | None = None
    slide_number: int | None = None
    row_start: int | None = None
    row_end: int | None = None
    section_path: str | None = None
    other_details: str = ""
    other_details_format: str = "kvpipe"
    schema_version: str = "v1"


EXAMPLE_SOURCE_DOCUMENT = SourceDocument(
    doc_id="doc_001",
    source_name="employee_handbook.pdf",
    source_uri="/data/docs/employee_handbook.pdf",
    source_kind="pdf",
    mime_type="application/pdf",
    file_ext="pdf",
    corpus_id="hr_docs",
    tenant_id="tenant_001",
    version="1",
    content_hash="sha256:abc123",
    timestamp_added=datetime.now(timezone.utc),
    timestamp_modified=datetime.now(timezone.utc),
    other_details="department:hr|||confidential:yes|||",
    other_details_format="kvpipe",
    schema_version="v1",
    doc_summary="Company policies and employee guidelines.",
    language="en",
    token_count=1024,
    char_count=5600,
)


EXAMPLE_DOCUMENT_CHUNK = DocumentChunk(
    doc_id="doc_001",
    chunk_id="chunk_001",
    chunk_index=0,
    chunk_text="All employees must complete annual compliance training.",
    chunk_summary="Annual compliance training is mandatory.",
    text_vector=[0.1, 0.2, 0.3],
    summary_vector=[0.4, 0.5, 0.6],
    entities=["employees", "compliance training"],
    keywords=["training", "policy", "mandatory"],
    source_name="employee_handbook.pdf",
    source_uri="/data/docs/employee_handbook.pdf",
    source_kind="pdf",
    corpus_id="hr_docs",
    tenant_id="tenant_001",
    content_hash="sha256:abc123",
    chunk_hash="sha256:def456",
    timestamp_added=datetime.now(timezone.utc),
    timestamp_modified=datetime.now(timezone.utc),
    ingested_at=datetime.now(timezone.utc),
    language="en",
    token_count=64,
    char_count=350,
    embedding_model="nomic-embed-text",
    embedding_version="1",
    embedding_dim=768,
    page_number=12,
    other_details="table_extracted:false|||ocr_used:false|||",
    other_details_format="kvpipe",
    schema_version="v1",
)

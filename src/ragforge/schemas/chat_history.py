from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ChatSession(BaseModel):
    session_id: str
    conversation_id: str
    session_name: str | None = None
    tenant_id: str | None = None
    corpus_id: str
    source_kind: str = "chat"
    started_at: datetime
    last_active_at: datetime
    timestamp_added: datetime
    timestamp_modified: datetime
    other_details: str = ""
    other_details_format: str = "kvpipe"
    schema_version: str = "v1"


class ChatTurn(BaseModel):
    session_id: str
    conversation_id: str
    turn_id: str
    message_id: str
    turn_index: int
    message_index: int | None = None
    role: str
    content: str
    summary: str | None = None
    text_vector: list[float] | None = None
    summary_vector: list[float] | None = None
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
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
    reply_to_message_id: str | None = None
    other_details: str = ""
    other_details_format: str = "kvpipe"
    schema_version: str = "v1"


EXAMPLE_CHAT_SESSION = ChatSession(
    session_id="sess_001",
    conversation_id="conv_001",
    session_name="Support chat",
    tenant_id="tenant_001",
    corpus_id="chat_history",
    source_kind="chat",
    started_at=datetime.now(timezone.utc),
    last_active_at=datetime.now(timezone.utc),
    timestamp_added=datetime.now(timezone.utc),
    timestamp_modified=datetime.now(timezone.utc),
    other_details="channel:web|||product:ragsystem|||",
    other_details_format="kvpipe",
    schema_version="v1",
)


EXAMPLE_CHAT_TURN = ChatTurn(
    session_id="sess_001",
    conversation_id="conv_001",
    turn_id="turn_001",
    message_id="msg_001",
    turn_index=0,
    role="user",
    content="How do I reset my password?",
    summary="User asks about password reset.",
    text_vector=[0.1, 0.2, 0.3],
    summary_vector=[0.4, 0.5, 0.6],
    entities=["password reset"],
    keywords=["reset", "password"],
    corpus_id="chat_history",
    tenant_id="tenant_001",
    content_hash="sha256:ghi789",
    chunk_hash="sha256:jkl012",
    timestamp_added=datetime.now(timezone.utc),
    timestamp_modified=datetime.now(timezone.utc),
    ingested_at=datetime.now(timezone.utc),
    language="en",
    token_count=12,
    char_count=29,
    embedding_model="nomic-embed-text",
    embedding_version="1",
    embedding_dim=768,
    reply_to_message_id=None,
    other_details="channel:web|||intent:account_help|||",
    other_details_format="kvpipe",
    schema_version="v1",
)

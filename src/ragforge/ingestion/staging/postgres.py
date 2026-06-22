from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ragforge.config import INGESTION_STAGE_POSTGRES_URL, INGESTION_STAGE_SCHEMA
from ragforge.logging import get_logger
from ragforge.ingestion.staging.models import (
    StagedChunkRecord,
    StagedPageBlock,
    StagedSourceDocument,
)


class PostgresIngestionStore:
    def __init__(
        self, connection_url: str | None = None, schema_name: str | None = None
    ):
        self.connection_url = connection_url or INGESTION_STAGE_POSTGRES_URL
        self.schema_name = schema_name or INGESTION_STAGE_SCHEMA
        self.logger = get_logger("ragforge.ingestion.staging.postgres")
        self._init_db()

    def _get_connection(self):
        import psycopg2
        from psycopg2.extras import RealDictCursor

        return psycopg2.connect(self.connection_url, cursor_factory=RealDictCursor)

    def _init_db(self):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name};")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.schema_name}.source_documents (
                        doc_id TEXT PRIMARY KEY,
                        logical_key TEXT NOT NULL,
                        source_name TEXT NOT NULL,
                        source_uri TEXT NOT NULL,
                        source_kind TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        version TEXT,
                        status TEXT NOT NULL DEFAULT 'active',
                        payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        timestamp_added TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        timestamp_modified TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    """)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.schema_name}.page_blocks (
                        block_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL,
                        page_num INTEGER,
                        loading_strategy TEXT NOT NULL,
                        payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        timestamp_added TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    """)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.schema_name}.chunk_records (
                        chunk_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        timestamp_added TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    """)
                conn.commit()
        except Exception as exc:
            conn.rollback()
            raise RuntimeError(
                f"Failed to initialize ingestion staging DB: {exc}"
            ) from exc
        finally:
            conn.close()

    def register_source_document(self, record: StagedSourceDocument) -> str:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT doc_id, content_hash, status
                    FROM {self.schema_name}.source_documents
                    WHERE logical_key = %s AND status = 'active'
                    ORDER BY timestamp_modified DESC
                    LIMIT 1;
                    """,
                    (record.logical_key,),
                )
                existing = cur.fetchone()
                if existing and existing["content_hash"] == record.content_hash:
                    cur.execute(
                        f"""
                        UPDATE {self.schema_name}.source_documents
                        SET timestamp_modified = CURRENT_TIMESTAMP
                        WHERE doc_id = %s;
                        """,
                        (existing["doc_id"],),
                    )
                    conn.commit()
                    return "unchanged"
                if existing and existing["content_hash"] != record.content_hash:
                    cur.execute(
                        f"""
                        UPDATE {self.schema_name}.source_documents
                        SET status = 'superseded', timestamp_modified = CURRENT_TIMESTAMP
                        WHERE doc_id = %s;
                        """,
                        (existing["doc_id"],),
                    )
                cur.execute(
                    f"""
                    INSERT INTO {self.schema_name}.source_documents
                    (doc_id, logical_key, source_name, source_uri, source_kind, content_hash, version, status, payload, timestamp_added, timestamp_modified)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE
                    SET logical_key = EXCLUDED.logical_key,
                        source_name = EXCLUDED.source_name,
                        source_uri = EXCLUDED.source_uri,
                        source_kind = EXCLUDED.source_kind,
                        content_hash = EXCLUDED.content_hash,
                        version = EXCLUDED.version,
                        status = EXCLUDED.status,
                        payload = EXCLUDED.payload,
                        timestamp_modified = EXCLUDED.timestamp_modified;
                    """,
                    (
                        record.doc_id,
                        record.logical_key,
                        record.source_name,
                        record.source_uri,
                        record.source_kind,
                        record.content_hash,
                        str(record.version) if record.version is not None else None,
                        record.status,
                        json.dumps(record.payload),
                        record.timestamp_added,
                        record.timestamp_modified,
                    ),
                )
                conn.commit()
                return "updated" if existing else "new"
        except Exception as exc:
            conn.rollback()
            raise RuntimeError(f"Failed to register source document: {exc}") from exc
        finally:
            conn.close()

    def save_page_block(self, record: StagedPageBlock) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.schema_name}.page_blocks
                    (block_id, doc_id, page_num, loading_strategy, payload, timestamp_added)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                    ON CONFLICT (block_id) DO UPDATE
                    SET payload = EXCLUDED.payload,
                        loading_strategy = EXCLUDED.loading_strategy;
                    """,
                    (
                        record.block_id,
                        record.doc_id,
                        record.page_num,
                        record.loading_strategy,
                        json.dumps(record.payload),
                        record.timestamp_added,
                    ),
                )
                conn.commit()
        except Exception as exc:
            conn.rollback()
            raise RuntimeError(f"Failed to save page block: {exc}") from exc
        finally:
            conn.close()

    def save_chunk_record(self, record: StagedChunkRecord) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.schema_name}.chunk_records
                    (chunk_id, doc_id, chunk_index, payload, timestamp_added)
                    VALUES (%s, %s, %s, %s::jsonb, %s)
                    ON CONFLICT (chunk_id) DO UPDATE
                    SET payload = EXCLUDED.payload;
                    """,
                    (
                        record.chunk_id,
                        record.doc_id,
                        record.chunk_index,
                        json.dumps(record.payload),
                        record.timestamp_added,
                    ),
                )
                conn.commit()
        except Exception as exc:
            conn.rollback()
            raise RuntimeError(f"Failed to save chunk record: {exc}") from exc
        finally:
            conn.close()

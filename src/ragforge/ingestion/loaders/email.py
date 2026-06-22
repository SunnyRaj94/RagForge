from __future__ import annotations

import asyncio
import email
from email import policy
from email.message import Message
from pathlib import Path

from ragforge.ingestion.loaders.base import BaseDocumentLoader, register_loader
from ragforge.ingestion.models import DocumentUnit, LoadedBlock
from ragforge.ingestion.processors import BaseUnitProcessor
from ragforge.ingestion.utils import hash_file, normalize_whitespace


def _message_text(msg: Message) -> str:
    if msg.is_multipart():
        parts = []
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                continue
            if part.get_content_type() == "text/plain":
                payload = part.get_content()
                if payload:
                    parts.append(str(payload))
        return "\n".join(parts)
    payload = msg.get_content()
    return str(payload) if payload else ""


@register_loader(".eml")
class EmailLoader(BaseUnitProcessor, BaseDocumentLoader):
    def _build_units(
        self, file_path: str, config_path: str | None = None
    ) -> list[DocumentUnit]:
        raw = Path(file_path).read_bytes()
        msg = email.message_from_bytes(raw, policy=policy.default)
        file_hash = hash_file(file_path)
        units: list[DocumentUnit] = []

        body_text = normalize_whitespace(_message_text(msg))
        if body_text:
            units.append(
                DocumentUnit(
                    doc_id=file_hash,
                    source_path=file_path,
                    source_name=Path(file_path).name,
                    file_type="email",
                    unit_num=1,
                    raw_text=body_text,
                    metadata={
                        "source": file_path,
                        "filename": Path(file_path).name,
                        "file_type": "email",
                        "subject": msg.get("subject", ""),
                        "from": msg.get("from", ""),
                        "to": msg.get("to", ""),
                        "message_id": msg.get("message-id", ""),
                    },
                    unit_kind="email_body",
                )
            )
        return units

    def _load_blocks(
        self, file_path: str, config_path: str | None = None
    ) -> list[LoadedBlock]:
        from ragforge.ingestion.loaders.attachment import AttachmentLoader

        units = self._build_units(file_path, config_path=config_path)
        body_blocks = self.process_units(units, config_path=config_path)

        raw = Path(file_path).read_bytes()
        msg = email.message_from_bytes(raw, policy=policy.default)
        attachment_loader = AttachmentLoader()
        attachment_blocks: list[LoadedBlock] = []
        attachment_count = 0
        for part in msg.walk():
            if part.get_content_disposition() != "attachment":
                continue
            attachment_count += 1
            filename = part.get_filename() or f"attachment-{attachment_count}"
            payload = part.get_payload(decode=True) or b""
            attachment_blocks.extend(
                attachment_loader.load_bytes(
                    filename,
                    payload,
                    content_type=part.get_content_type(),
                    parent_metadata={
                        "source": file_path,
                        "filename": Path(file_path).name,
                        "file_type": "email",
                        "subject": msg.get("subject", ""),
                    },
                    config_path=config_path,
                )
            )

        return body_blocks + attachment_blocks

    def load(self, file_path: str, config_path: str | None = None) -> list[LoadedBlock]:
        return self._load_blocks(file_path, config_path=config_path)

    async def load_async(
        self, file_path: str, config_path: str | None = None
    ) -> list[LoadedBlock]:
        return await asyncio.to_thread(self._load_blocks, file_path, config_path)

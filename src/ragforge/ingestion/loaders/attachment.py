from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from ragforge.ingestion.loaders.base import BaseDocumentLoader, loader_for_path
from ragforge.ingestion.models import LoadedBlock


class AttachmentLoader(BaseDocumentLoader):
    def load(self, file_path: str, config_path: str | None = None) -> list[LoadedBlock]:
        return loader_for_path(file_path).load(file_path, config_path=config_path)

    def load_bytes(
        self,
        file_name: str,
        data: bytes,
        *,
        content_type: str | None = None,
        parent_metadata: dict[str, Any] | None = None,
        config_path: str | None = None,
    ) -> list[LoadedBlock]:
        suffix = Path(file_name).suffix or ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fh:
            fh.write(data)
            temp_path = fh.name
        try:
            blocks = loader_for_path(temp_path).load(temp_path, config_path=config_path)
            for block in blocks:
                block.metadata.update(
                    {
                        "attachment_name": file_name,
                        "attachment_content_type": content_type,
                        "parent_metadata": parent_metadata or {},
                    }
                )
            return blocks
        finally:
            Path(temp_path).unlink(missing_ok=True)

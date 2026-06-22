from __future__ import annotations
from typing import Any
from ragforge.ingestion.loaders.attachment import AttachmentLoader
from ragforge.ingestion.loaders.base import BaseDocumentLoader, loader_for_path
from ragforge.ingestion.loaders.excel import ExcelLoader
from ragforge.ingestion.loaders.email import EmailLoader
from ragforge.ingestion.loaders.image import ImageLoader
from ragforge.ingestion.loaders.pdf import PdfLoader
from ragforge.ingestion.loaders.pptx import PptxLoader
from ragforge.ingestion.loaders.text import TextLoader


def load_file(file_path: str, config_path: str | None = None) -> list[dict[str, Any]]:
    loader = loader_for_path(file_path)
    return [
        block.to_legacy_dict()
        for block in loader.load(file_path, config_path=config_path)
    ]


def load_attachment(
    file_name: str,
    data: bytes,
    *,
    content_type: str | None = None,
    parent_metadata: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> list[dict[str, Any]]:
    loader = AttachmentLoader()
    return [
        block.to_legacy_dict()
        for block in loader.load_bytes(
            file_name,
            data,
            content_type=content_type,
            parent_metadata=parent_metadata or {},
            config_path=config_path,
        )
    ]


__all__ = [
    "BaseDocumentLoader",
    "PdfLoader",
    "ExcelLoader",
    "PptxLoader",
    "TextLoader",
    "ImageLoader",
    "EmailLoader",
    "AttachmentLoader",
    "load_file",
    "load_attachment",
]

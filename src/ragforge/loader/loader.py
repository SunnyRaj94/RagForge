from __future__ import annotations

import os
from typing import Any

import yaml

from ragforge.ingestion.loaders import (
    load_attachment as _load_attachment,
    load_file as _load_file,
)


def load_chunking_config(config_path: str | None = None) -> dict[str, Any]:
    if config_path is None:
        from ragforge.config import get_config_path

        config_path = get_config_path()
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_pdf(file_path: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    return _load_file(file_path)


def parse_excel(file_path: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    return _load_file(file_path)


def parse_pptx(file_path: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    return _load_file(file_path)


def parse_image(file_path: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    return _load_file(file_path)


def parse_email(file_path: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    return _load_file(file_path)


def parse_attachment(
    file_name: str,
    data: bytes,
    *,
    content_type: str | None = None,
    parent_metadata: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> list[dict[str, Any]]:
    return _load_attachment(
        file_name,
        data,
        content_type=content_type,
        parent_metadata=parent_metadata,
        config_path=config_path,
    )


def load_file_legacy(
    file_path: str, config_path: str | None = None
) -> list[dict[str, Any]]:
    return _load_file(file_path, config_path=config_path)


# Backward-compatible alias used by existing imports.
load_file = load_file_legacy

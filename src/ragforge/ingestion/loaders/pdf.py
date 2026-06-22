from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import fitz

from ragforge.ingestion.loaders.base import BaseDocumentLoader, register_loader
from ragforge.ingestion.models import DocumentUnit, LoadedBlock
from ragforge.ingestion.processors import BaseUnitProcessor
from ragforge.ingestion.routing import route_loaded_block
from ragforge.ingestion.utils import hash_file, normalize_whitespace


def _load_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        from ragforge.config import get_config_path

        config_path = get_config_path()
    if not os.path.exists(config_path):
        return {}
    import yaml

    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@register_loader(".pdf")
class PdfLoader(BaseUnitProcessor, BaseDocumentLoader):
    def _build_units(
        self, file_path: str, config_path: str | None = None
    ) -> list[DocumentUnit]:
        config = _load_config(config_path)
        pdf_config = config.get("pdf", {})
        extract_tables = pdf_config.get("extract_tables", True)
        ocr_fallback = pdf_config.get("ocr_fallback", True)

        doc = fitz.open(file_path)
        units: list[DocumentUnit] = []
        file_hash = hash_file(file_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            digital_text = normalize_whitespace(page.get_text())
            tables: list[dict[str, Any]] = []
            if extract_tables:
                try:
                    detected_tables = page.find_tables()
                    for idx, table in enumerate(detected_tables):
                        cells = table.extract()
                        if cells:
                            tables.append(
                                {
                                    "table_index": idx,
                                    "cells": cells,
                                    "rows": len(cells),
                                }
                            )
                except Exception:
                    pass

            image_count = len(page.get_images(full=True))
            route = route_loaded_block(
                digital_text,
                has_tables=bool(tables),
                image_count=image_count,
                ocr_enabled=ocr_fallback,
            ).loading_strategy

            ocr_bytes = None
            if route in {"ocr", "llm"} and ocr_fallback:
                pix = page.get_pixmap(dpi=150)
                ocr_bytes = pix.tobytes("png")

            units.append(
                DocumentUnit(
                    doc_id=file_hash,
                    source_path=file_path,
                    source_name=Path(file_path).name,
                    file_type="pdf",
                    unit_num=page_num + 1,
                    raw_text=digital_text,
                    tables=tables,
                    image_count=image_count,
                    metadata={
                        "source": file_path,
                        "filename": Path(file_path).name,
                        "file_type": "pdf",
                        "page": page_num + 1,
                        "ocr_run": route in {"ocr", "llm"},
                        "table_count": len(tables),
                        "image_count": image_count,
                    },
                    loading_strategy_hint=route,
                    ocr_image_bytes=ocr_bytes,
                    ocr_image_name=f"{Path(file_path).stem}_page_{page_num + 1}.png",
                    unit_kind="page",
                )
            )

        doc.close()
        return units

    def load(self, file_path: str, config_path: str | None = None) -> list[LoadedBlock]:
        units = self._build_units(file_path, config_path=config_path)
        return self.process_units(units, config_path=config_path)

    async def load_async(
        self, file_path: str, config_path: str | None = None
    ) -> list[LoadedBlock]:
        units = self._build_units(file_path, config_path=config_path)
        return await self.process_units_async(units, config_path=config_path)

from __future__ import annotations
from ragforge.logging import get_logger
import asyncio
import os
import tempfile
from typing import Any

import easyocr
import httpx

from ragforge.config import DEFAULT_LLM_MODEL, OLLAMA_URL
from ragforge.ingestion.models import DocumentUnit, LoadedBlock, PageContent
from ragforge.ingestion.routing import route_loaded_block
from ragforge.ingestion.utils import (
    build_retrieval_text,
    extract_important_terms,
    normalize_whitespace,
    summarize_text,
)

_OCR_READER = None
logger = get_logger(__name__)


class BaseUnitProcessor:
    def _load_config(self, config_path: str | None = None) -> dict[str, Any]:
        if not config_path:
            from ragforge.config import get_config_path

            logger.debug("No config path provided, using default from config module.")
            config_path = get_config_path()
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}")
            return {}
        import yaml

        logger.debug(f"Loading config from {config_path}")
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _get_ocr_reader(self, languages: list[str] | None = None):
        global _OCR_READER
        if _OCR_READER is None:
            try:
                _OCR_READER = easyocr.Reader(languages or ["en"], gpu=True)
            except Exception:
                _OCR_READER = easyocr.Reader(languages or ["en"], gpu=False)
        return _OCR_READER

    def _ocr_from_bytes(
        self,
        image_bytes: bytes,
        *,
        languages: list[str] | None = None,
        suffix: str = ".png",
    ) -> str:
        logger.debug("Performing OCR on image bytes")
        if not image_bytes:
            return ""
        reader = self._get_ocr_reader(languages)
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as fh:
            fh.write(image_bytes)
            fh.flush()
            result = reader.readtext(fh.name)
        return normalize_whitespace(" ".join(item[1] for item in result))

    def _llm_summarize(
        self,
        text: str,
        *,
        model_id: str | None = None,
        max_chars: int = 1500,
    ) -> str:
        text = normalize_whitespace(text)
        if not text:
            return ""
        model = model_id or DEFAULT_LLM_MODEL
        prompt = (
            "Summarize the following document chunk in 2-4 concise sentences.\n\n"
            f"{text[:max_chars]}"
        )
        logger.debug(f"Summarizing text with model '{model}' (max {max_chars} chars)")
        try:
            response = httpx.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60.0,
            )
            response.raise_for_status()
            payload = response.json()
            summary = payload.get("response", "")
            logger.debug(f"LLM summary generated (length {len(summary)} chars)")
            return normalize_whitespace(summary) or summarize_text(text)
        except Exception:
            logger.exception("LLM summarization failed, falling back to basic summary")
            return summarize_text(text)

    def _table_to_markdown(self, cells: list[list[Any]]) -> str:
        if not cells or not any(cells):
            return ""
        max_cols = max(len(row) for row in cells if row)
        lines: list[str] = []
        header = cells[0] if cells else []
        header_str = "| " + " | ".join(
            str(cell or "").replace("\n", " ") for cell in header
        )
        if len(header) < max_cols:
            header_str += " | " + " | ".join("" for _ in range(max_cols - len(header)))
        lines.append(header_str + " |")
        lines.append("| " + " | ".join("---" for _ in range(max_cols)) + " |")
        for row in cells[1:]:
            if not row:
                continue
            row_str = "| " + " | ".join(
                str(cell or "").replace("\n", " ") for cell in row
            )
            if len(row) < max_cols:
                row_str += " | " + " | ".join("" for _ in range(max_cols - len(row)))
            lines.append(row_str + " |")
        return "\n".join(lines)

    def process_unit(
        self, unit: DocumentUnit, config_path: str | None = None
    ) -> LoadedBlock:
        config = self._load_config(config_path)
        shared_cfg = config.get("ingestion", {})
        file_cfg = config.get(unit.file_type, {})
        ocr_enabled = bool(shared_cfg.get("ocr_enabled", True))
        llm_enabled = bool(shared_cfg.get("llm_enabled", True))
        ocr_languages = shared_cfg.get(
            "ocr_languages", file_cfg.get("ocr_language", ["en"])
        )
        llm_model = shared_cfg.get("llm_model", DEFAULT_LLM_MODEL)

        route = (
            unit.loading_strategy_hint
            or route_loaded_block(
                unit.raw_text,
                has_tables=bool(unit.tables),
                image_count=unit.image_count,
                ocr_enabled=ocr_enabled,
            ).loading_strategy
        )

        ocr_text = ""
        if route in {"ocr", "llm"} and ocr_enabled and unit.ocr_image_bytes:
            ocr_text = self._ocr_from_bytes(
                unit.ocr_image_bytes, languages=ocr_languages
            )

        table_text = ""
        tables_as_markdown: list[dict[str, Any]] = []
        for idx, table in enumerate(unit.tables):
            markdown = table.get("markdown")
            if not markdown and isinstance(table.get("cells"), list):
                markdown = self._table_to_markdown(table["cells"])
            if markdown:
                tables_as_markdown.append(
                    {
                        **table,
                        "table_index": table.get("table_index", idx),
                        "markdown": markdown,
                    }
                )
                table_text += ("\n\n" if table_text else "") + markdown

        base_text = normalize_whitespace(
            " ".join(part for part in [unit.raw_text, ocr_text, table_text] if part)
        )
        if not base_text and not tables_as_markdown and not ocr_text:
            return LoadedBlock(
                doc_id=unit.doc_id,
                page_num=unit.unit_num,
                loading_strategy=route,
                metadata={
                    **unit.metadata,
                    "source": unit.source_path,
                    "filename": unit.source_name,
                    "file_type": unit.file_type,
                    "loading_strategy": route,
                },
                content=PageContent(
                    tables=tables_as_markdown,
                    text="",
                    images=[],
                    ocr_text=ocr_text or None,
                    llm_summary=None,
                    important_terms=[],
                    numbers=[],
                ),
                text="",
            )

        llm_summary = None
        if route == "llm" and llm_enabled:
            llm_summary = self._llm_summarize(base_text, model_id=llm_model)

        summary = llm_summary or summarize_text(base_text)
        important_terms = extract_important_terms(base_text)
        retrieval_text = build_retrieval_text(summary, important_terms, base_text)
        metadata = {
            **unit.metadata,
            "source": unit.source_path,
            "filename": unit.source_name,
            "file_type": unit.file_type,
            "loading_strategy": route,
            "ocr_run": bool(ocr_text),
            "table_count": len(tables_as_markdown),
            "image_count": unit.image_count,
            "unit_kind": unit.unit_kind,
            "unit_num": unit.unit_num,
        }
        return LoadedBlock(
            doc_id=unit.doc_id,
            page_num=unit.unit_num,
            loading_strategy=route,
            metadata=metadata,
            content=PageContent(
                tables=tables_as_markdown,
                text=base_text,
                images=[],
                ocr_text=ocr_text or None,
                llm_summary=llm_summary,
                important_terms=important_terms,
                numbers=[],
            ),
            text=retrieval_text,
        )

    def process_units(
        self, units: list[DocumentUnit], config_path: str | None = None
    ) -> list[LoadedBlock]:
        return [self.process_unit(unit, config_path=config_path) for unit in units]

    async def process_units_async(
        self, units: list[DocumentUnit], config_path: str | None = None
    ) -> list[LoadedBlock]:
        tasks = [
            asyncio.to_thread(self.process_unit, unit, config_path) for unit in units
        ]
        return list(await asyncio.gather(*tasks))

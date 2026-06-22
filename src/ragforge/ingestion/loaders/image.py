from __future__ import annotations

from pathlib import Path

from ragforge.ingestion.loaders.base import BaseDocumentLoader, register_loader
from ragforge.ingestion.models import DocumentUnit, LoadedBlock
from ragforge.ingestion.processors import BaseUnitProcessor
from ragforge.ingestion.utils import hash_file


@register_loader(".png", ".jpg", ".jpeg", ".bmp", ".tiff")
class ImageLoader(BaseUnitProcessor, BaseDocumentLoader):
    def _build_units(self, file_path: str) -> list[DocumentUnit]:
        file_hash = hash_file(file_path)
        return [
            DocumentUnit(
                doc_id=file_hash,
                source_path=file_path,
                source_name=Path(file_path).name,
                file_type="image",
                unit_num=1,
                raw_text="",
                image_count=1,
                metadata={"source": file_path, "filename": Path(file_path).name},
                loading_strategy_hint="ocr",
                ocr_image_bytes=Path(file_path).read_bytes(),
                ocr_image_name=Path(file_path).name,
                unit_kind="image",
            )
        ]

    def load(self, file_path: str, config_path: str | None = None) -> list[LoadedBlock]:
        units = self._build_units(file_path)
        return self.process_units(units, config_path=config_path)

    async def load_async(
        self, file_path: str, config_path: str | None = None
    ) -> list[LoadedBlock]:
        units = self._build_units(file_path)
        return await self.process_units_async(units, config_path=config_path)

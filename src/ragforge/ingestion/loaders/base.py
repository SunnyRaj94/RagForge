from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from ragforge.ingestion.models import LoadedBlock

LOADER_REGISTRY: dict[str, type["BaseDocumentLoader"]] = {}


def register_loader(*extensions: str):
    def decorator(cls: type["BaseDocumentLoader"]):
        for ext in extensions:
            LOADER_REGISTRY[ext.lower()] = cls
        return cls

    return decorator


def loader_for_path(file_path: str) -> "BaseDocumentLoader":
    ext = Path(file_path).suffix.lower()
    loader_cls = LOADER_REGISTRY.get(
        ext, LOADER_REGISTRY.get("*", BaseDocumentLoaderFallback)
    )
    return loader_cls()


class BaseDocumentLoader(ABC):
    supported_extensions: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def load(self, file_path: str, config_path: str | None = None) -> list[LoadedBlock]:
        raise NotImplementedError

    async def load_async(
        self, file_path: str, config_path: str | None = None
    ) -> list[LoadedBlock]:
        return await asyncio.to_thread(self.load, file_path, config_path)


class BaseDocumentLoaderFallback(BaseDocumentLoader):
    def load(self, file_path: str, config_path: str | None = None) -> list[LoadedBlock]:
        from ragforge.ingestion.loaders.text import TextLoader

        return TextLoader().load(file_path, config_path=config_path)

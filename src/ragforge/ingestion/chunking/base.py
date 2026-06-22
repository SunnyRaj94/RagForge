from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseChunker(ABC):
    @abstractmethod
    def chunk(
        self, documents: list[dict[str, Any]], config_path: str | None = None
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

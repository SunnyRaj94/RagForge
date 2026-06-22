from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int = 768) -> str:
        raise NotImplementedError

    @abstractmethod
    def upsert_documents(
        self, collection_name: str, documents: list[dict[str, Any]]
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def search_documents(
        self, collection_name: str, query: str, limit: int = 5, **kwargs: Any
    ) -> str:
        raise NotImplementedError

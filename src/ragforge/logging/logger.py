from __future__ import annotations

import json
import logging
from typing import Any

_LOGGER_INITIALIZED = False


class _StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any]
        if isinstance(record.msg, dict):
            payload = record.msg
        else:
            payload = {"message": record.getMessage()}
        payload.setdefault("logger", record.name)
        payload.setdefault("level", record.levelname)
        return json.dumps(payload, ensure_ascii=False)


def _configure_root_logger() -> None:
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return
    root = logging.getLogger("ragforge")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_StructuredFormatter())
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    _LOGGER_INITIALIZED = True


def get_logger(name: str, **context: Any) -> logging.LoggerAdapter:
    _configure_root_logger()
    base_logger = logging.getLogger(name)
    return logging.LoggerAdapter(base_logger, context)

from __future__ import annotations

from typing import Any

PAIR_SEPARATOR = "|||"
KEY_VALUE_SEPARATOR = ":"


def _escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(PAIR_SEPARATOR, "\\|||")
        .replace(KEY_VALUE_SEPARATOR, "\\:")
    )


def _unescape(value: str) -> str:
    return (
        value.replace("\\:", KEY_VALUE_SEPARATOR)
        .replace("\\|||", PAIR_SEPARATOR)
        .replace("\\\\", "\\")
    )


def serialize_kvpipe(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in data.items():
        parts.append(f"{_escape(str(key))}{KEY_VALUE_SEPARATOR}{_escape(str(value))}")
    return PAIR_SEPARATOR.join(parts) + (PAIR_SEPARATOR if parts else "")


def deserialize_kvpipe(text: str) -> dict[str, str]:
    if not text:
        return {}
    result: dict[str, str] = {}
    for part in text.split(PAIR_SEPARATOR):
        if not part:
            continue
        if KEY_VALUE_SEPARATOR not in part:
            continue
        key, value = part.split(KEY_VALUE_SEPARATOR, 1)
        result[_unescape(key)] = _unescape(value)
    return result

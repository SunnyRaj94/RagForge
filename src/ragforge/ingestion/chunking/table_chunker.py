from __future__ import annotations


def split_table_markdown(text: str, chunk_size: int) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        current.append(line)
        if len("\n".join(current)) >= chunk_size:
            chunks.append("\n".join(current).strip())
            current = current[-2:] if len(current) > 2 else current
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]

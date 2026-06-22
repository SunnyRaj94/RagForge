from __future__ import annotations

from ragforge.ingestion.utils import split_into_sentences


def split_text_with_overlap(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence)
        if current and current_len + sentence_len > chunk_size:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)
            overlap_sentences = current[-2:] if chunk_overlap > 0 else []
            current = list(overlap_sentences)
            current_len = sum(len(x) for x in current)
        current.append(sentence)
        current_len += sentence_len
    if current:
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)
    return chunks or [text]

from __future__ import annotations

import re
from collections import Counter

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "not",
    "but",
    "you",
    "your",
    "our",
    "out",
    "all",
    "can",
    "will",
    "into",
    "than",
    "then",
    "there",
    "what",
    "when",
    "where",
    "who",
    "why",
    "how",
    "about",
    "also",
    "more",
    "some",
    "such",
    "only",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def summarize_text(text: str, max_chars: int = 240) -> str:
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    sentence_end = max(text.find("."), text.find("!"), text.find("?"))
    if 0 < sentence_end < max_chars:
        return text[: sentence_end + 1]
    return text[:max_chars].rstrip()


def extract_important_terms(text: str, max_terms: int = 12) -> list[str]:
    normalized = normalize_whitespace(text).lower()
    tokens = re.findall(r"[a-zA-Z0-9$%][a-zA-Z0-9$%./-]*", normalized)
    numeric_tokens = [tok for tok in tokens if any(ch.isdigit() for ch in tok)]
    keyword_tokens = [
        tok
        for tok in tokens
        if len(tok) >= 6 and tok not in STOPWORDS and not tok.isdigit()
    ]
    acronym_tokens = [
        tok
        for tok in re.findall(r"\b[A-Z]{2,}\b", text)
        if tok.lower() not in STOPWORDS
    ]
    ranked = numeric_tokens + acronym_tokens + keyword_tokens
    counts = Counter(ranked)
    return [term for term, _ in counts.most_common(max_terms)]


def build_retrieval_text(summary: str, terms: list[str], raw_text: str) -> str:
    parts = [
        normalize_whitespace(summary),
        ",".join(terms),
        normalize_whitespace(raw_text),
    ]
    return "|||".join(parts)


def split_into_sentences(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

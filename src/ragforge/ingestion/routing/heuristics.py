from __future__ import annotations

from ragforge.ingestion.models import RoutingDecision
from ragforge.ingestion.utils.text_normalization import normalize_whitespace


def route_loaded_block(
    text: str,
    *,
    has_tables: bool = False,
    image_count: int = 0,
    ocr_enabled: bool = True,
) -> RoutingDecision:
    normalized = normalize_whitespace(text)
    text_length = len(normalized)
    text_density = text_length / max(1, image_count + 1)

    if not normalized and image_count > 0 and ocr_enabled:
        return RoutingDecision(
            loading_strategy="ocr",
            reason="No digital text found and images are present",
            confidence=0.95,
        )

    if has_tables and text_length > 0:
        return RoutingDecision(
            loading_strategy="table_text",
            reason="Tables detected with extractable text",
            confidence=0.88,
        )

    if image_count > 0 and text_density < 120:
        return RoutingDecision(
            loading_strategy="llm",
            reason="Low text density with image content suggests semantic summarization",
            confidence=0.72,
        )

    if normalized:
        return RoutingDecision(
            loading_strategy="text",
            reason="Digital text is available",
            confidence=0.99,
        )

    return RoutingDecision(
        loading_strategy="ocr" if ocr_enabled else "text",
        reason="Fallback strategy selected",
        confidence=0.5,
    )

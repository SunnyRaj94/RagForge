from typing import List, Dict, Any
from src.ragforge.loader.loader import load_chunking_config
from src.ragforge.utils.logging_hooks import observe_stage


def split_text_by_chars(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Splits a long text string into overlapping chunks based on character count.
    Splits are made gracefully on spaces/newlines if possible.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # If we are not at the end of the text, try to find a space or newline to split on
        if end < len(text):
            # Search backwards for a space or newline within the last 15% of the window
            lookback = int(chunk_size * 0.15)
            split_idx = -1
            for i in range(end, end - lookback, -1):
                if text[i] in [" ", "\n"]:
                    split_idx = i
                    break
            if split_idx != -1:
                end = split_idx

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap
        if start >= len(text) - chunk_overlap:
            break

    return chunks


@observe_stage("chunk_documents")
def chunk_documents(
    documents: List[Dict[Any, Any]], config_path: str = None
) -> List[Dict[str, Any]]:
    """
    Takes parsed document blocks and splits them further if they exceed configured chunk limits.
    """
    config = load_chunking_config(config_path)
    default_config = config.get("default", {"chunk_size": 800, "chunk_overlap": 100})

    chunked_docs = []

    for doc in documents:
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        file_type = metadata.get("file_type", "default")

        # Get chunking strategy for this file type
        type_config = config.get(file_type, default_config)
        chunk_size = type_config.get(
            "chunk_size", default_config.get("chunk_size", 800)
        )
        chunk_overlap = type_config.get(
            "chunk_overlap", default_config.get("chunk_overlap", 100)
        )

        # For Excel and PPTX, slide/batch structure is preserved.
        # Only split if it exceeds chunk size.
        should_split = file_type in ["pdf", "text", "default"] or len(text) > chunk_size

        if should_split:
            sub_chunks = split_text_by_chars(text, chunk_size, chunk_overlap)
            for idx, sub_text in enumerate(sub_chunks):
                new_meta = metadata.copy()
                new_meta["chunk_index"] = idx
                chunked_docs.append({"text": sub_text, "metadata": new_meta})
        else:
            chunked_docs.append(doc)

    return chunked_docs

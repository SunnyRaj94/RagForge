import pytest
from qdrant_client import QdrantClient
from servers.qdrant_mcp import (
    create_collection,
    upsert_documents,
    search_documents,
    get_embeddings,
)


def test_embeddings():
    """Verify that Ollama connection is active and can generate embedding vectors."""
    try:
        vector = get_embeddings("Hello World")
        assert isinstance(vector, list)
        assert len(vector) == 768
    except Exception as e:
        pytest.fail(f"Ollama embedding generation failed: {e}")


def test_qdrant_flow():
    """Test full create -> upsert -> search cycle on Qdrant using MCP server tools."""
    collection_name = "test_mcp_collection"
    client = QdrantClient(url="http://localhost:6333")

    # 1. Clean up collection if it exists
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    # 2. Test create collection
    res_create = create_collection(collection_name, 768)
    assert "created successfully" in res_create

    # 3. Test upsert
    test_docs = [
        {
            "id": "doc1",
            "text": "Temporal orchestrates workflow state machines reliably.",
            "metadata": {"topic": "temporal"},
        },
        {
            "id": "doc2",
            "text": "Qdrant is a fast vector database written in Rust.",
            "metadata": {"topic": "qdrant"},
        },
    ]
    res_upsert = upsert_documents(collection_name, test_docs)
    assert "Successfully upserted" in res_upsert

    # 4. Test search
    res_search = search_documents(
        collection_name, "rust vector database query", limit=1
    )
    assert "Qdrant" in res_search
    assert "Score:" in res_search
    assert "Lineage:" in res_search

    # Test search with a session_id matches globally ingested documents (lacking session_id)
    res_search_session = search_documents(
        collection_name, "rust vector database query", session_id="test-session-xyz", limit=1
    )
    assert "Qdrant" in res_search_session

    # Clean up
    client.delete_collection(collection_name)


if __name__ == "__main__":
    import sys

    # Run tests directly if executed as main script
    test_embeddings()
    test_qdrant_flow()
    print("All Qdrant MCP server tests passed successfully!")

import pytest
from src.ragforge.agents.rag.agent import RagForgeAgent


def test_rag_agent_initialization():
    """Verify that RagForgeAgent can be initialized and config attributes are stored correctly."""
    agent = RagForgeAgent(llm_model="gemma4:e4b", session_id="test-session-123")
    assert agent.llm_model == "gemma4:e4b"
    assert agent.session_id == "test-session-123"
    assert agent.tool_server_map == {}
    assert agent.raw_tools == []
    assert agent._graph is None


def test_rag_agent_graph_builder():
    """Verify that building the graph works when raw tools are populated."""
    agent = RagForgeAgent(llm_model="gemma4:e4b", session_id="test-session-123")
    # Manually populate raw tools to mock tool discovery
    agent.raw_tools = [
        {
            "name": "search_documents",
            "description": "Search Qdrant",
            "schema": {},
            "server": "qdrant",
        },
        {
            "name": "create_project_task",
            "description": "Create OP task",
            "schema": {},
            "server": "openproject",
        },
    ]
    agent.tool_server_map = {
        "search_documents": "qdrant",
        "create_project_task": "openproject",
    }

    graph = agent.build_graph()
    assert graph is not None
    assert agent.graph is not None

    # Verify tool metadata helpers
    assert agent.get_tool_server_map() == agent.tool_server_map
    assert agent.get_all_tool_names() == ["search_documents", "create_project_task"]
    assert "create_project_task" in agent.get_write_tools()

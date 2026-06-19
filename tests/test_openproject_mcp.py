import pytest
from servers.openproject_mcp import get_project_list, get_project_tasks


def test_get_project_list():
    """Verify that get_project_list fetches real projects from local OpenProject container."""
    try:
        res = get_project_list()
        assert "ID:" in res
        assert "Demo project" in res
        assert "Scrum project" in res
    except Exception as e:
        pytest.fail(f"get_project_list failed: {e}")


def test_get_project_tasks():
    """Verify that get_project_tasks fetches active tasks for Demo project (ID: 1)."""
    try:
        res = get_project_tasks("1")
        assert "ID:" in res
        # Check that some tasks or a message is returned
        assert isinstance(res, str)
        assert len(res) > 0
    except Exception as e:
        pytest.fail(f"get_project_tasks failed: {e}")


if __name__ == "__main__":
    import sys

    test_get_project_list()
    test_get_project_tasks()
    print("All OpenProject MCP server read tests passed successfully!")

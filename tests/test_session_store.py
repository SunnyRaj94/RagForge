import pytest
import shutil
import tempfile
from pathlib import Path
from src.ragforge.session_store import JSONSessionStore


def test_json_session_store():
    # Create a temp directory for JSON sessions
    temp_dir = tempfile.mkdtemp()
    try:
        store = JSONSessionStore(temp_dir)
        session_id = "test-session-123"
        name = "Test Session Name"
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        # Save session
        store.save_session(session_id, name, messages)

        # List sessions
        sessions = store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == session_id
        assert sessions[0]["name"] == name

        # Load session
        loaded = store.load_session(session_id)
        assert loaded["session_id"] == session_id
        assert loaded["name"] == name
        assert loaded["messages"] == messages

        # Delete session
        store.delete_session(session_id)
        sessions_after = store.list_sessions()
        assert len(sessions_after) == 0

    finally:
        shutil.rmtree(temp_dir)


def test_postgres_session_store():
    from src.ragforge.config import SESSION_STORE_POSTGRES_URL
    from src.ragforge.session_store import PostgresSessionStore

    session_id = "test-pg-session-999"
    name = "Test Postgres Session"
    messages = [
        {"role": "user", "content": "is postgres session working?"},
        {"role": "assistant", "content": "yes it is!"},
    ]

    store = PostgresSessionStore(SESSION_STORE_POSTGRES_URL)

    # Clean up first in case it remained from a dirty run
    store.delete_session(session_id)

    try:
        # Save session
        store.save_session(session_id, name, messages)

        # List sessions
        sessions = store.list_sessions()
        matching = [s for s in sessions if s["session_id"] == session_id]
        assert len(matching) == 1
        assert matching[0]["name"] == name

        # Load session
        loaded = store.load_session(session_id)
        assert loaded["session_id"] == session_id
        assert loaded["name"] == name
        assert loaded["messages"] == messages

        # Overwrite test
        new_messages = messages + [{"role": "user", "content": "awesome"}]
        store.save_session(session_id, name, new_messages)
        loaded = store.load_session(session_id)
        assert len(loaded["messages"]) == 3

    finally:
        # Delete session
        store.delete_session(session_id)
        sessions = store.list_sessions()
        matching = [s for s in sessions if s["session_id"] == session_id]
        assert len(matching) == 0

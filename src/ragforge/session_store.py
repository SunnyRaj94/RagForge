import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime


class BaseSessionStore(ABC):
    @abstractmethod
    def save_session(
        self, session_id: str, name: str, messages: List[Dict[str, Any]]
    ) -> None:
        """Save or update a session with its messages."""
        pass

    @abstractmethod
    def load_session(self, session_id: str) -> Dict[str, Any]:
        """Load session details including messages."""
        pass

    @abstractmethod
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all saved sessions."""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Delete a session by ID."""
        pass


class JSONSessionStore(BaseSessionStore):
    def __init__(self, json_dir: str):
        self.json_dir = Path(json_dir)
        self.json_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, session_id: str) -> Path:
        return self.json_dir / f"{session_id}.json"

    def save_session(
        self, session_id: str, name: str, messages: List[Dict[str, Any]]
    ) -> None:
        filepath = self._get_path(session_id)

        created_at = datetime.utcnow().isoformat()
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    created_at = data.get("created_at", created_at)
            except Exception:
                pass

        data = {
            "session_id": session_id,
            "name": name,
            "created_at": created_at,
            "updated_at": datetime.utcnow().isoformat(),
            "messages": messages,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_session(self, session_id: str) -> Dict[str, Any]:
        filepath = self._get_path(session_id)
        if not filepath.exists():
            return {
                "session_id": session_id,
                "name": f"Session {session_id}",
                "messages": [],
                "created_at": datetime.utcnow().isoformat(),
            }
        with open(filepath, "r") as f:
            return json.load(f)

    def list_sessions(self) -> List[Dict[str, Any]]:
        sessions = []
        for file in self.json_dir.glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    sessions.append(
                        {
                            "session_id": data.get("session_id"),
                            "name": data.get("name"),
                            "created_at": data.get("created_at"),
                            "updated_at": data.get("updated_at"),
                        }
                    )
            except Exception:
                pass
        # Sort by updated_at or created_at descending
        sessions.sort(
            key=lambda s: s.get("updated_at", s.get("created_at", "")), reverse=True
        )
        return sessions

    def delete_session(self, session_id: str) -> None:
        filepath = self._get_path(session_id)
        if filepath.exists():
            filepath.unlink()


class PostgresSessionStore(BaseSessionStore):
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self._init_db()

    def _get_connection(self):
        import psycopg2
        from psycopg2.extras import RealDictCursor

        return psycopg2.connect(self.connection_url, cursor_factory=RealDictCursor)

    def _init_db(self):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id VARCHAR(100) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        messages JSONB NOT NULL
                    );
                """)
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to initialize PostgresSessionStore: {str(e)}")
        finally:
            conn.close()

    def save_session(
        self, session_id: str, name: str, messages: List[Dict[str, Any]]
    ) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_sessions (session_id, name, messages, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (session_id) DO UPDATE
                    SET name = EXCLUDED.name,
                        messages = EXCLUDED.messages,
                        updated_at = CURRENT_TIMESTAMP;
                """,
                    (session_id, name, json.dumps(messages)),
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def load_session(self, session_id: str) -> Dict[str, Any]:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT session_id, name, messages, created_at FROM chat_sessions WHERE session_id = %s",
                    (session_id,),
                )
                row = cur.fetchone()
                if row:
                    return {
                        "session_id": row["session_id"],
                        "name": row["name"],
                        "messages": row["messages"],
                        "created_at": row["created_at"].isoformat(),
                    }
                else:
                    return {
                        "session_id": session_id,
                        "name": f"Session {session_id}",
                        "messages": [],
                        "created_at": datetime.utcnow().isoformat(),
                    }
        finally:
            conn.close()

    def list_sessions(self) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT session_id, name, created_at, updated_at FROM chat_sessions ORDER BY updated_at DESC"
                )
                rows = cur.fetchall()
                sessions = []
                for row in rows:
                    sessions.append(
                        {
                            "session_id": row["session_id"],
                            "name": row["name"],
                            "created_at": row["created_at"].isoformat(),
                            "updated_at": row["updated_at"].isoformat(),
                        }
                    )
                return sessions
        finally:
            conn.close()

    def delete_session(self, session_id: str) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chat_sessions WHERE session_id = %s", (session_id,)
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()


def get_session_store() -> BaseSessionStore:
    from ragforge.config import (
        SESSION_STORE_TYPE,
        SESSION_STORE_JSON_DIR,
        SESSION_STORE_POSTGRES_URL,
        ROOT_DIR,
    )

    if SESSION_STORE_TYPE.lower() == "postgres":
        return PostgresSessionStore(SESSION_STORE_POSTGRES_URL)
    else:
        # Default to JSON
        json_path = Path(SESSION_STORE_JSON_DIR)
        if not json_path.is_absolute():
            json_path = (ROOT_DIR / json_path).resolve()
        return JSONSessionStore(str(json_path))

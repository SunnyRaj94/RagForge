import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# Locate config.yaml in workspace root
ROOT_DIR = Path(__file__).parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"

_config_data = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r") as f:
        _config_data = yaml.safe_load(f) or {}

general_cfg = _config_data.get("general", {})

OLLAMA_URL = os.getenv(
    "OLLAMA_URL", general_cfg.get("ollama_url", "http://localhost:11434")
)
QDRANT_URL = os.getenv(
    "QDRANT_URL", general_cfg.get("qdrant_url", "http://localhost:6333")
)
OPENPROJECT_URL = os.getenv(
    "OPENPROJECT_URL", general_cfg.get("openproject_url", "http://localhost:8080")
)
OPENPROJECT_API_KEY = os.getenv(
    "OPENPROJECT_API_KEY", general_cfg.get("openproject_api_key", "")
)
TEMPORAL_URL = os.getenv(
    "TEMPORAL_URL", general_cfg.get("temporal_url", "localhost:7233")
)
PHOENIX_COLLECTOR_URL = os.getenv(
    "PHOENIX_COLLECTOR_URL",
    general_cfg.get("phoenix_collector_url", "http://localhost:6006/v1/traces"),
)
MLFLOW_TRACKING_URI = general_cfg.get("mlflow_tracking_uri") or os.getenv(
    "MLFLOW_TRACKING_URI", "file:./mlruns"
)

DEFAULT_EMBEDDING_MODEL = general_cfg.get("default_embedding_model", "nomic-embed-text")
DEFAULT_LLM_MODEL = general_cfg.get("default_llm_model", "gemma4:e4b")
DEFAULT_COLLECTION = os.getenv(
    "DEFAULT_COLLECTION", general_cfg.get("default_collection", "ragforge-collection")
)

# Session Store Configurations
session_store_cfg = _config_data.get("session_store", {})
SESSION_STORE_TYPE = os.getenv(
    "SESSION_STORE_TYPE", session_store_cfg.get("type", "json")
)
SESSION_STORE_POSTGRES_URL = os.getenv(
    "SESSION_STORE_POSTGRES_URL",
    session_store_cfg.get(
        "postgres_url", "postgresql://temporal:temporal@localhost:5432/temporal"
    ),
)
SESSION_STORE_JSON_DIR = os.getenv(
    "SESSION_STORE_JSON_DIR", session_store_cfg.get("json_dir", "./data/sessions")
)
CHAT_HISTORY_COLLECTION = os.getenv(
    "CHAT_HISTORY_COLLECTION",
    session_store_cfg.get("chat_history_collection", "chat-history-collection"),
)
ROLLING_WINDOW_TURNS = int(
    os.getenv(
        "ROLLING_WINDOW_TURNS",
        str(session_store_cfg.get("rolling_window_turns", 3)),
    )
)


def get_config_path() -> str:
    return str(CONFIG_PATH)

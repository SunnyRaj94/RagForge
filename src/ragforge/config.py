import os
import yaml
from pathlib import Path

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
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    general_cfg.get("mlflow_tracking_uri", "http://localhost:5000"),
)

DEFAULT_EMBEDDING_MODEL = general_cfg.get("default_embedding_model", "nomic-embed-text")
DEFAULT_LLM_MODEL = general_cfg.get("default_llm_model", "gemma4:e4b")


def get_config_path() -> str:
    return str(CONFIG_PATH)

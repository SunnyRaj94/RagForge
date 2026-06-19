import os
from datetime import timedelta
from typing import List, Dict, Any
from temporalio import workflow, activity


# Activities for Ingestion Workflow
@activity.defn
async def scan_directory_activity(directory_path: str) -> List[str]:
    """Scans a directory for supported document types."""
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Path '{directory_path}' does not exist.")

    if os.path.isfile(directory_path):
        return [directory_path]

    supported_extensions = {
        ".pdf",
        ".xlsx",
        ".xls",
        ".pptx",
        ".ppt",
        ".png",
        ".jpg",
        ".jpeg",
        ".txt",
    }
    file_paths = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                file_paths.append(os.path.join(root, file))

    return file_paths


@activity.defn
async def parse_and_chunk_activity(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Loads and chunks a document file."""
    # Local imports to avoid Temporal sandbox validation errors
    from src.ragforge.loader.loader import load_file
    from src.ragforge.chunking.chunker import chunk_documents

    file_path = payload["file_path"]
    config_path = payload.get("config_path", None)

    # Load raw documents
    raw_docs = load_file(file_path, config_path=config_path)
    # Chunk them
    chunked_docs = chunk_documents(raw_docs, config_path=config_path)
    return chunked_docs


@activity.defn
async def embed_and_index_activity(payload: Dict[str, Any]) -> str:
    """Creates a collection and indexes the chunks into Qdrant."""
    # Local imports to avoid Temporal sandbox validation errors
    from src.ragforge.index.indexer import create_collection, upsert_documents

    collection_name = payload["collection_name"]
    chunks = payload["chunks"]
    session_id = payload.get("session_id", None)

    # Inject session_id if provided
    if session_id:
        for chunk in chunks:
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            chunk["metadata"]["session_id"] = session_id

    # 1. Create collection (default 768 dimensions for nomic-embed-text)
    create_collection(collection_name, 768)

    # 2. Upsert
    res = upsert_documents(collection_name, chunks)
    return res


@activity.defn
async def log_mlflow_run_activity(payload: Dict[str, Any]) -> str:
    """Logs the ingestion execution to MLflow."""
    import mlflow
    from src.ragforge.config import MLFLOW_TRACKING_URI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ragforge-ingestion")

    with mlflow.start_run() as run:
        mlflow.log_param("collection_name", payload["collection_name"])
        mlflow.log_param("total_files_processed", payload["total_files"])
        mlflow.log_metric("total_chunks_indexed", payload["total_chunks"])
        mlflow.log_metric("execution_time_seconds", payload["execution_time"])

    return f"MLflow Run logged. Run ID: {run.info.run_id}"


# Ingestion Workflow Definition - Unsandboxed
@workflow.defn(sandboxed=False)
class IngestionWorkflow:
    @workflow.run
    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinates document ingestion:
        1. Scan directory
        2. Parse & chunk files
        3. Index into Qdrant
        4. Log to MLflow
        """
        directory_path = payload["directory_path"]
        collection_name = payload["collection_name"]
        config_path = payload.get("config_path", None)
        session_id = payload.get("session_id", None)

        start_time = workflow.now().timestamp()

        # 1. Scan directory
        file_paths = await workflow.execute_activity(
            scan_directory_activity,
            directory_path,
            start_to_close_timeout=timedelta(minutes=5),
        )

        if not file_paths:
            return {
                "status": "SUCCESS",
                "message": "No files found to process.",
                "total_files": 0,
                "total_chunks": 0,
            }

        all_chunks = []

        # 2. Parse & chunk files
        for file_path in file_paths:
            chunks = await workflow.execute_activity(
                parse_and_chunk_activity,
                {"file_path": file_path, "config_path": config_path},
                start_to_close_timeout=timedelta(minutes=10),
            )
            all_chunks.extend(chunks)

        # 3. Index in Qdrant
        index_result = ""
        if all_chunks:
            index_result = await workflow.execute_activity(
                embed_and_index_activity,
                {
                    "collection_name": collection_name,
                    "chunks": all_chunks,
                    "session_id": session_id,
                },
                start_to_close_timeout=timedelta(minutes=15),
            )

        execution_time = workflow.now().timestamp() - start_time

        # 4. Log to MLflow
        mlflow_result = await workflow.execute_activity(
            log_mlflow_run_activity,
            {
                "collection_name": collection_name,
                "total_files": len(file_paths),
                "total_chunks": len(all_chunks),
                "execution_time": execution_time,
            },
            start_to_close_timeout=timedelta(minutes=2),
        )

        return {
            "status": "SUCCESS",
            "message": f"Ingestion completed. {index_result}",
            "total_files": len(file_paths),
            "total_chunks": len(all_chunks),
            "execution_time_seconds": execution_time,
            "mlflow": mlflow_result,
        }

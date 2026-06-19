import pytest
import uuid
import asyncio
from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.worker import Worker

# Import workflows and activities
from src.ragforge.workflows.ingestion import (
    IngestionWorkflow,
    scan_directory_activity,
    parse_and_chunk_activity,
    embed_and_index_activity,
    log_mlflow_run_activity,
)
from src.ragforge.workflows.openproject import (
    OpenProjectWriteWorkflow,
    create_work_package_activity,
    update_work_package_status_activity,
    add_work_package_comment_activity,
)

load_dotenv()


@pytest.mark.anyio
async def test_workflows_end_to_end():
    # 1. Connect Client
    client = await Client.connect("localhost:7233")
    task_queue = f"test-queue-{uuid.uuid4()}"

    # 2. Start Worker for test queue
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[IngestionWorkflow, OpenProjectWriteWorkflow],
        activities=[
            scan_directory_activity,
            parse_and_chunk_activity,
            embed_and_index_activity,
            log_mlflow_run_activity,
            create_work_package_activity,
            update_work_package_status_activity,
            add_work_package_comment_activity,
        ],
    )

    # Run worker in background
    worker_task = asyncio.create_task(worker.run())

    try:
        # Test OpenProjectWriteWorkflow - Create
        result_create = await client.execute_workflow(
            OpenProjectWriteWorkflow.run,
            {
                "action": "create",
                "project_id": "1",
                "title": "Temporal Test Workflow Task",
                "description": "Task created from workflow test.",
                "task_type": "Task",
            },
            id=f"test-op-create-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        assert result_create["id"] is not None
        assert result_create["subject"] == "Temporal Test Workflow Task"
        task_id = result_create["id"]

        # Test OpenProjectWriteWorkflow - Update Status
        result_update = await client.execute_workflow(
            OpenProjectWriteWorkflow.run,
            {
                "action": "update_status",
                "task_id": str(task_id),
                "status_name": "In progress",
            },
            id=f"test-op-update-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        assert result_update["id"] == task_id
        assert result_update["status"] == "In progress"

        # Test OpenProjectWriteWorkflow - Add Comment
        result_comment = await client.execute_workflow(
            OpenProjectWriteWorkflow.run,
            {
                "action": "add_comment",
                "task_id": str(task_id),
                "comment_text": "Workflow test comment.",
            },
            id=f"test-op-comment-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        assert result_comment["id"] is not None
        assert "Workflow test comment." in result_comment["comment"]

        # Test IngestionWorkflow
        import os

        test_file = f"test_workflow_ingest_{uuid.uuid4()}.txt"
        with open(test_file, "w") as f:
            f.write("RagForge + Temporal RAG Pipeline Integration Test.")

        try:
            result_ingestion = await client.execute_workflow(
                IngestionWorkflow.run,
                {
                    "directory_path": test_file,
                    "collection_name": f"test_ingestion_coll_{str(uuid.uuid4())[:8]}",
                },
                id=f"test-ingest-{uuid.uuid4()}",
                task_queue=task_queue,
            )
            assert result_ingestion["status"] == "SUCCESS"
            assert result_ingestion["total_files"] == 1
            assert result_ingestion["total_chunks"] == 1
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    finally:
        # Stop worker
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

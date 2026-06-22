import os
import asyncio
from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.worker import Worker

# Load workflows and activities
from ragforge.workflows.ingestion import (
    IngestionWorkflow,
    scan_directory_activity,
    parse_and_chunk_activity,
    embed_and_index_activity,
    log_mlflow_run_activity,
)
from ragforge.workflows.openproject import (
    OpenProjectWriteWorkflow,
    create_work_package_activity,
    update_work_package_status_activity,
    add_work_package_comment_activity,
)

load_dotenv()


async def main():
    from ragforge.config import TEMPORAL_URL

    temporal_url = TEMPORAL_URL

    # Connect to Temporal Server
    client = await Client.connect(temporal_url)
    print(f"Temporal worker connected to server at {temporal_url}")

    # Define Worker
    worker = Worker(
        client,
        task_queue="ragforge-tasks",
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

    print("Starting worker on queue 'ragforge-tasks'...")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())

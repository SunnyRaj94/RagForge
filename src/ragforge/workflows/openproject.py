import os
import httpx
from datetime import timedelta
from typing import Dict, Any
from temporalio import workflow, activity
from temporalio.common import RetryPolicy

# Define type and status mappings
TYPE_MAP = {
    "task": 1,
    "milestone": 2,
    "phase": 3,
    "feature": 4,
    "epic": 5,
    "user story": 6,
    "bug": 7,
}

STATUS_MAP = {
    "new": 1,
    "in specification": 2,
    "specified": 3,
    "confirmed": 4,
    "to be scheduled": 5,
    "scheduled": 6,
    "in progress": 7,
    "developed": 8,
    "in testing": 9,
    "tested": 10,
    "test failed": 11,
    "closed": 12,
    "on hold": 13,
    "rejected": 14,
}


def _get_auth_headers():
    from src.ragforge.config import OPENPROJECT_URL, OPENPROJECT_API_KEY

    return OPENPROJECT_URL, ("apikey", OPENPROJECT_API_KEY)


@activity.defn
async def create_work_package_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a work package in OpenProject."""
    url, auth = _get_auth_headers()
    project_id = payload["project_id"]
    title = payload["title"]
    description = payload["description"]
    task_type = payload.get("task_type", "task").lower()

    type_id = TYPE_MAP.get(task_type, 1)

    post_payload = {
        "subject": title,
        "description": {"format": "markdown", "raw": description},
        "_links": {"type": {"href": f"/api/v3/types/{type_id}"}},
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{url}/api/v3/projects/{project_id}/work_packages",
            json=post_payload,
            auth=auth,
            timeout=30.0,
        )
        response.raise_for_status()
        res_json = response.json()
        return {
            "id": res_json["id"],
            "subject": res_json["subject"],
            "type": res_json["_links"]["type"]["title"],
            "status": res_json["_links"]["status"]["title"],
        }


@activity.defn
async def update_work_package_status_activity(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Updates the status of a work package. Handles fetching lockVersion first."""
    url, auth = _get_auth_headers()
    task_id = payload["task_id"]
    status_name = payload["status_name"].lower()

    # 1. Fetch live status list from OpenProject to match name dynamically
    status_id = STATUS_MAP.get(status_name)

    async with httpx.AsyncClient() as client:
        if not status_id:
            # Fallback dynamic lookup
            res_statuses = await client.get(
                f"{url}/api/v3/statuses", auth=auth, timeout=10.0
            )
            res_statuses.raise_for_status()
            elements = res_statuses.json().get("_embedded", {}).get("elements", [])
            for st in elements:
                if st["name"].lower() == status_name:
                    status_id = st["id"]
                    break
            if not status_id:
                raise ValueError(f"Status '{status_name}' not found in OpenProject.")

        # 2. Get current work package to retrieve lockVersion
        res_wp = await client.get(
            f"{url}/api/v3/work_packages/{task_id}", auth=auth, timeout=15.0
        )
        res_wp.raise_for_status()
        wp_json = res_wp.json()
        lock_version = wp_json["lockVersion"]

        # 3. Patch the status
        patch_payload = {
            "lockVersion": lock_version,
            "_links": {"status": {"href": f"/api/v3/statuses/{status_id}"}},
        }
        response = await client.patch(
            f"{url}/api/v3/work_packages/{task_id}",
            json=patch_payload,
            auth=auth,
            timeout=30.0,
        )
        response.raise_for_status()
        res_json = response.json()
        return {
            "id": res_json["id"],
            "subject": res_json["subject"],
            "status": res_json["_links"]["status"]["title"],
        }


@activity.defn
async def add_work_package_comment_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Adds a comment to an existing work package."""
    url, auth = _get_auth_headers()
    task_id = payload["task_id"]
    comment_text = payload["comment_text"]

    post_payload = {"comment": {"format": "markdown", "raw": comment_text}}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{url}/api/v3/work_packages/{task_id}/activities",
            json=post_payload,
            auth=auth,
            timeout=30.0,
        )
        response.raise_for_status()
        res_json = response.json()
        return {"id": res_json["id"], "comment": res_json["comment"]["raw"]}


# OpenProject Write Workflow
@workflow.defn(sandboxed=False)
class OpenProjectWriteWorkflow:
    @workflow.run
    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes OpenProject write actions with retry support.
        """
        action = payload["action"]

        # Configure exponential backoff retry policy for robustness
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=2),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(seconds=30),
            maximum_attempts=0,  # Infinite retries for write actions
        )

        if action == "create":
            return await workflow.execute_activity(
                create_work_package_activity,
                payload,
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=retry_policy,
            )
        elif action == "update_status":
            return await workflow.execute_activity(
                update_work_package_status_activity,
                payload,
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=retry_policy,
            )
        elif action == "add_comment":
            return await workflow.execute_activity(
                add_work_package_comment_activity,
                payload,
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=retry_policy,
            )
        else:
            raise ValueError(f"Unknown action '{action}' for OpenProjectWriteWorkflow")

from mcp.server.fastmcp import FastMCP
import httpx
import os
import uuid
from dotenv import load_dotenv
from temporalio.client import Client

load_dotenv()

# Create FastMCP server
mcp = FastMCP("openproject-server")

from src.ragforge.config import OPENPROJECT_URL, OPENPROJECT_API_KEY, TEMPORAL_URL

# Connect to OpenProject
openproject_url = OPENPROJECT_URL
openproject_api_key = OPENPROJECT_API_KEY

# Connect to Temporal
temporal_url = TEMPORAL_URL


def _get_auth():
    return ("apikey", openproject_api_key)


@mcp.tool()
def get_project_list() -> str:
    """
    Retrieve the list of projects from OpenProject.
    Returns names, IDs, and identifiers of the projects.
    """
    try:
        url = f"{openproject_url}/api/v3/projects"
        response = httpx.get(url, auth=_get_auth())
        response.raise_for_status()

        elements = response.json().get("_embedded", {}).get("elements", [])
        if not elements:
            return "No projects found."

        lines = []
        for p in elements:
            lines.append(
                f"- ID: {p['id']} | Name: {p['name']} | Identifier: {p['identifier']}"
            )

        return "\n".join(lines)
    except Exception as e:
        return f"Error retrieving projects: {str(e)}"


@mcp.tool()
def get_project_tasks(project_id: str) -> str:
    """
    Retrieve the active tasks (work packages) for a given project.
    Can accept numeric project ID or project identifier string.
    """
    try:
        url = f"{openproject_url}/api/v3/projects/{project_id}/work_packages"
        response = httpx.get(url, auth=_get_auth())
        response.raise_for_status()

        elements = response.json().get("_embedded", {}).get("elements", [])
        if not elements:
            return f"No tasks found for project '{project_id}'."

        lines = []
        for wp in elements:
            status = wp.get("_links", {}).get("status", {}).get("title", "N/A")
            task_type = wp.get("_links", {}).get("type", {}).get("title", "N/A")
            lines.append(
                f"- [ID: {wp['id']}] [{task_type}] {wp['subject']} (Status: {status})"
            )

        return "\n".join(lines)
    except Exception as e:
        return f"Error retrieving tasks for project '{project_id}': {str(e)}"


@mcp.tool()
async def create_project_task(
    project_id: str, title: str, description: str, task_type: str = "Task"
) -> str:
    """
    Create a new task in a project. This triggers a reliable Temporal workflow to write the task.
    Arguments:
    - project_id: The numeric ID or string identifier of the project.
    - title: Subject/Title of the task.
    - description: Description of the task in Markdown format.
    - task_type: Type of task, e.g. "Task", "Bug", "Feature", "Milestone", "Phase", "Epic", "User story".
    """
    try:
        # Connect to Temporal Client
        client = await Client.connect(temporal_url)

        workflow_id = f"op-create-task-{uuid.uuid4()}"

        result = await client.execute_workflow(
            "OpenProjectWriteWorkflow",
            arg={
                "action": "create",
                "project_id": project_id,
                "title": title,
                "description": description,
                "task_type": task_type,
            },
            id=workflow_id,
            task_queue="ragforge-tasks",
        )

        return f"Success: Task created in OpenProject via Temporal workflow. Workflow ID: {workflow_id}. Result: {result}"
    except Exception as e:
        return f"Error triggering workflow to create task: {str(e)}"


@mcp.tool()
async def update_task_status(task_id: str, status_name: str) -> str:
    """
    Update the status of an existing task (work package) by ID. Triggers a Temporal workflow.
    Arguments:
    - task_id: Numeric work package ID.
    - status_name: Target status name (e.g. "In progress", "Closed", "New").
    """
    try:
        client = await Client.connect(temporal_url)
        workflow_id = f"op-update-status-{uuid.uuid4()}"

        result = await client.execute_workflow(
            "OpenProjectWriteWorkflow",
            arg={
                "action": "update_status",
                "task_id": task_id,
                "status_name": status_name,
            },
            id=workflow_id,
            task_queue="ragforge-tasks",
        )

        return f"Success: Task status update requested via Temporal workflow. Workflow ID: {workflow_id}. Result: {result}"
    except Exception as e:
        return f"Error triggering workflow to update status: {str(e)}"


@mcp.tool()
async def add_task_comment(task_id: str, comment_text: str) -> str:
    """
    Add a comment to an existing task (work package) by ID. Triggers a Temporal workflow.
    Arguments:
    - task_id: Numeric work package ID.
    - comment_text: Text of the comment to add.
    """
    try:
        client = await Client.connect(temporal_url)
        workflow_id = f"op-add-comment-{uuid.uuid4()}"

        result = await client.execute_workflow(
            "OpenProjectWriteWorkflow",
            arg={
                "action": "add_comment",
                "task_id": task_id,
                "comment_text": comment_text,
            },
            id=workflow_id,
            task_queue="ragforge-tasks",
        )

        return f"Success: Task comment addition requested via Temporal workflow. Workflow ID: {workflow_id}. Result: {result}"
    except Exception as e:
        return f"Error triggering workflow to add comment: {str(e)}"


if __name__ == "__main__":
    mcp.run()

import os
import httpx
import asyncio
import uuid
import streamlit as st
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from temporalio.client import Client
from src.ragforge.config import (
    OLLAMA_URL,
    TEMPORAL_URL,
    OPENPROJECT_URL,
    OPENPROJECT_API_KEY,
    ROLLING_WINDOW_TURNS,
)
from src.ragforge.session_store import get_session_store


load_dotenv()

# Set up page configurations
st.set_page_config(
    page_title="RagForge RAG Agent Core",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    code, pre, [class*="mono"] {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    
    /* Card design */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    .metric-header {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6366f1;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
    }
    
    /* Agent thoughts */
    .thought-container {
        border-left: 4px solid #a855f7;
        background: rgba(168, 85, 247, 0.03);
        padding: 0.75rem 1rem;
        margin: 0.75rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .thought-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #a855f7;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    
    .observation-container {
        border-left: 4px solid #10b981;
        background: rgba(16, 185, 129, 0.03);
        padding: 0.75rem 1rem;
        margin: 0.75rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .observation-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #10b981;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar settings
st.sidebar.markdown(
    "<h2 style='font-weight: 800; color: #a855f7;'>Config & Ingestion</h2>",
    unsafe_allow_html=True,
)

# Fetch Ollama models dynamically
ollama_url = OLLAMA_URL
try:
    res = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
    models_data = res.json().get("models", [])
    # Filter out embedding models from LLM list (they do not support chat endpoint)
    model_names = [m["name"] for m in models_data if "embed" not in m["name"].lower()]
    if not model_names:
        model_names = ["gemma4:e4b", "llama3:latest", "qwen2.5-coder:latest"]
except Exception:
    model_names = ["gemma4:e4b", "llama3:latest", "qwen2.5-coder:latest"]

# Determine a responsive default model (prefer qwen2.5-coder:latest or llama3:latest if available)
default_idx = 0
for idx, name in enumerate(model_names):
    if "qwen2.5-coder" in name:
        default_idx = idx
        break
    elif "llama3" in name:
        default_idx = idx

selected_llm = st.sidebar.selectbox(
    "LLM Reasoning Model", model_names, index=default_idx
)

# Directory ingestion form
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<h3 style='font-weight: 600; font-size: 1.1rem;'>Trigger Ingestion</h3>",
    unsafe_allow_html=True,
)
ingest_path = st.sidebar.text_input("Source Path (File or Dir)", value="docs")
collection_name = st.sidebar.text_input(
    "Qdrant Collection", value="ragforge-collection"
)


async def run_ingestion_workflow(path: str, collection: str):
    temporal_url = TEMPORAL_URL
    client = await Client.connect(temporal_url)

    workflow_id = f"ingestion-{uuid.uuid4()}"

    handle = await client.start_workflow(
        "IngestionWorkflow",
        arg={"directory_path": path, "collection_name": collection},
        id=workflow_id,
        task_queue="ragforge-tasks",
    )
    return handle


if st.sidebar.button("Run Ingestion Pipeline", use_container_width=True):
    if not ingest_path:
        st.sidebar.error("Please provide a valid source path.")
    else:
        with st.sidebar.status("Ingesting files...", expanded=True) as status:
            try:
                handle = asyncio.run(
                    run_ingestion_workflow(ingest_path, collection_name)
                )
                status.write(
                    f"Ingestion started. Workflow Run ID: {handle.first_execution_run_id}"
                )

                # Wait for result
                result = asyncio.run(handle.result())
                status.update(
                    label="Ingestion Complete!", state="complete", expanded=False
                )
                st.sidebar.success(
                    f"Successfully processed {result.get('total_files', 0)} files ({result.get('total_chunks', 0)} chunks)."
                )
                st.sidebar.info(
                    f"Execution time: {result.get('execution_time_seconds', 0.0):.2f}s"
                )
                st.sidebar.caption(result.get("mlflow", ""))
            except Exception as e:
                status.update(label="Ingestion Failed", state="error")
                st.sidebar.error(str(e))


def get_python_interpreter() -> str:
    venv_python = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python"
    )
    if os.path.exists(venv_python):
        return venv_python
    import sys

    return sys.executable


# Setup sessions and RAG agent loop
async def run_agent_turn(prompt: str, llm_model: str, recent_history: list = None, session_id: str = None) -> str:
    python_bin = get_python_interpreter()
    app_dir = os.path.dirname(os.path.abspath(__file__))
    qdrant_script = os.path.join(app_dir, "servers", "qdrant_mcp.py")
    openproject_script = os.path.join(app_dir, "servers", "openproject_mcp.py")

    # 1. Connect to Qdrant MCP and OpenProject MCP
    qdrant_params = StdioServerParameters(command=python_bin, args=[qdrant_script])
    openproject_params = StdioServerParameters(
        command=python_bin, args=[openproject_script]
    )

    thought_placeholder = st.empty()
    thoughts = []

    async with (
        stdio_client(qdrant_params) as (read_q, write_q),
        stdio_client(openproject_params) as (read_op, write_op),
    ):
        async with (
            ClientSession(read_q, write_q) as session_q,
            ClientSession(read_op, write_op) as session_op,
        ):
            await session_q.initialize()
            await session_op.initialize()

            # List tools
            tools_q = await session_q.list_tools()
            tools_op = await session_op.list_tools()

            # Combine tools
            all_tools = []
            for t in tools_q.tools:
                all_tools.append(
                    {
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.inputSchema,
                        "server": "qdrant",
                    }
                )
            for t in tools_op.tools:
                all_tools.append(
                    {
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.inputSchema,
                        "server": "openproject",
                    }
                )

            # Construct tools description for prompt
            tools_desc = ""
            for t in all_tools:
                tools_desc += f"- **{t['name']}**: {t['description']}\n  Parameters schema: {t['input_schema']}\n\n"

            system_prompt = f"""You are an intelligent RAG project management assistant. You have access to local tools for searching the document knowledge base (Qdrant) and interacting with OpenProject.

Current Session ID: {session_id}
Use this Session ID when calling tools (like `search_documents`, `upsert_documents`, `search_chat_history`, or `ingest_file_or_directory`) if you want to index new documents/turns for this session or filter search results to this session.

Available tools:
{tools_desc}

Use the following format for your thought process:
Thought: your reasoning about what tool to call or what to do next.
Action: the name of the tool you wish to invoke, formatted as a JSON block. E.g.:
```json
{{
  "name": "tool_name",
  "arguments": {{
    "arg1": "val1"
  }}
}}
```
Observation: the output of the tool.

Repeat this cycle as needed. Once you have gathered all necessary information or completed the tasks, provide the final answer to the user starting with "Final Answer:".

IMPORTANT:
- Use search_knowledge_base to find documents, requirements, templates, or instructions.
- Live OpenProject information can be fetched using get_project_list and get_project_tasks.
- Modifying tasks (create, update, comment) should be done using create_project_task, update_task_status, or add_task_comment, which trigger Temporal workflows.
"""

            messages = [
                {"role": "system", "content": system_prompt},
            ]
            if recent_history:
                messages.extend(recent_history)
            messages.append({"role": "user", "content": prompt})

            # ReAct loop
            max_iterations = 6
            for iteration in range(max_iterations):
                # Call Ollama
                response = httpx.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": llm_model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": 0.0},
                    },
                    timeout=90.0,
                )
                response.raise_for_status()
                assistant_message = response.json()["message"]["content"]

                # Check for Action
                action_index = assistant_message.find("Action:")
                thought_text = assistant_message
                if action_index != -1:
                    thought_text = assistant_message[:action_index]

                # Print thought step
                clean_thought = thought_text.replace("Thought:", "").strip()
                if clean_thought:
                    thoughts.append(("thought", clean_thought))
                    with thought_placeholder.container():
                        for t_type, t_val in thoughts:
                            if t_type == "thought":
                                st.markdown(
                                    f"<div class='thought-container'><div class='thought-header'>Thought</div>{t_val}</div>",
                                    unsafe_allow_html=True,
                                )
                            elif t_type == "observation":
                                st.markdown(
                                    f"<div class='observation-container'><div class='observation-header'>Observation</div>{t_val}</div>",
                                    unsafe_allow_html=True,
                                )

                # Check if we should execute an action
                if action_index != -1:
                    action_part = assistant_message[action_index + 7 :].strip()

                    # Extract JSON block
                    json_start = action_part.find("{")
                    json_end = action_part.rfind("}") + 1

                    if json_start != -1 and json_end != -1:
                        json_str = action_part[json_start:json_end]
                        try:
                            import json

                            action_data = json.loads(json_str)
                            tool_name = action_data["name"]
                            tool_args = action_data.get("arguments", {})

                            # Find which server exposes this tool
                            target_server = None
                            for t in all_tools:
                                if t["name"] == tool_name:
                                    target_server = t["server"]
                                    break

                            thoughts.append(
                                (
                                    "thought",
                                    f"Calling tool `{tool_name}` with args: `{tool_args}`",
                                )
                            )
                            with thought_placeholder.container():
                                for t_type, t_val in thoughts:
                                    if t_type == "thought":
                                        st.markdown(
                                            f"<div class='thought-container'><div class='thought-header'>Thought</div>{t_val}</div>",
                                            unsafe_allow_html=True,
                                        )
                                    elif t_type == "observation":
                                        st.markdown(
                                            f"<div class='observation-container'><div class='observation-header'>Observation</div>{t_val}</div>",
                                            unsafe_allow_html=True,
                                        )

                            # Execute tool
                            if target_server == "qdrant":
                                obs_res = await session_q.call_tool(
                                    tool_name, tool_args
                                )
                            elif target_server == "openproject":
                                obs_res = await session_op.call_tool(
                                    tool_name, tool_args
                                )
                            else:
                                raise ValueError(
                                    f"Tool {tool_name} not found on any MCP server."
                                )

                            obs_text = (
                                obs_res.content[0].text
                                if obs_res.content
                                else "No output."
                            )

                            # Log observation
                            thoughts.append(("observation", obs_text))
                            with thought_placeholder.container():
                                for t_type, t_val in thoughts:
                                    if t_type == "thought":
                                        st.markdown(
                                            f"<div class='thought-container'><div class='thought-header'>Thought</div>{t_val}</div>",
                                            unsafe_allow_html=True,
                                        )
                                    elif t_type == "observation":
                                        st.markdown(
                                            f"<div class='observation-container'><div class='observation-header'>Observation</div>{t_val}</div>",
                                            unsafe_allow_html=True,
                                        )

                            # Add to messages
                            messages.append(
                                {"role": "assistant", "content": assistant_message}
                            )
                            messages.append(
                                {"role": "user", "content": f"Observation: {obs_text}"}
                            )

                        except Exception as parse_err:
                            obs_text = f"Error executing tool action: {str(parse_err)}"
                            messages.append(
                                {"role": "assistant", "content": assistant_message}
                            )
                            messages.append(
                                {"role": "user", "content": f"Observation: {obs_text}"}
                            )
                    else:
                        obs_text = "Failed to parse action JSON payload."
                        messages.append(
                            {"role": "assistant", "content": assistant_message}
                        )
                        messages.append(
                            {"role": "user", "content": f"Observation: {obs_text}"}
                        )
                elif "Final Answer:" in assistant_message:
                    final_ans = assistant_message[
                        assistant_message.find("Final Answer:") + 13 :
                    ].strip()
                    thought_placeholder.empty()  # Clear step thinking output
                    return final_ans, thoughts
                else:
                    # Fallback final answer
                    thought_placeholder.empty()
                    return assistant_message, thoughts

            thought_placeholder.empty()
            return (
                "RAG Agent exceeded maximum thinking iterations without returning a final answer.",
                thoughts,
            )


# Main Layout
st.markdown("<h1 class='main-title'>RagForge RAG Chat</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Manage OpenProject tasks and search Qdrant knowledge bases using local ReAct Agent & custom MCP tools</div>",
    unsafe_allow_html=True,
)

# Metrics Grid for OpenProject Overview
st.markdown(
    "<h3 style='margin-bottom: 1rem; font-weight: 600;'>System Overview</h3>",
    unsafe_allow_html=True,
)
m_col1, m_col2, m_col3 = st.columns(3)

# Retrieve project/task summary dynamically
url = OPENPROJECT_URL
key = OPENPROJECT_API_KEY
try:
    auth = ("apikey", key)
    res_p = httpx.get(f"{url}/api/v3/projects", auth=auth, timeout=3.0)
    p_count = res_p.json().get("total", 0)

    # We can retrieve total tasks (work packages) across all projects
    res_wp = httpx.get(f"{url}/api/v3/work_packages", auth=auth, timeout=3.0)
    wp_count = res_wp.json().get("total", 0)

    # Get active status name
    status_msg = "Connected"
except Exception:
    p_count = "N/A"
    wp_count = "N/A"
    status_msg = "Offline"

with m_col1:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-header'>OpenProject API Status</div>
            <div class='metric-value' style='color: {"#10b981" if status_msg == "Connected" else "#ef4444"};'>{status_msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m_col2:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-header'>Active Projects</div>
            <div class='metric-value'>{p_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m_col3:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-header'>Total Work Packages</div>
            <div class='metric-value'>{wp_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Session store helper
async def index_chat_turn_async(session_id: str, user_msg: str, assistant_msg: str):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from datetime import datetime
    from src.ragforge.config import QDRANT_URL, DEFAULT_EMBEDDING_MODEL, CHAT_HISTORY_COLLECTION
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": DEFAULT_EMBEDDING_MODEL, "prompt": f"User: {user_msg}\nAssistant: {assistant_msg}"},
                timeout=30.0,
            )
            response.raise_for_status()
            vector = response.json()["embedding"]
        
        q_client = QdrantClient(url=QDRANT_URL)
        if not q_client.collection_exists(CHAT_HISTORY_COLLECTION):
            q_client.create_collection(
                collection_name=CHAT_HISTORY_COLLECTION,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        
        doc_id = str(uuid.uuid4())
        payload = {
            "text": f"User: {user_msg}\nAssistant: {assistant_msg}",
            "session_id": session_id,
            "user_message": user_msg,
            "assistant_message": assistant_msg,
            "timestamp": datetime.utcnow().isoformat()
        }
        q_client.upsert(
            collection_name=CHAT_HISTORY_COLLECTION,
            points=[PointStruct(id=doc_id, vector=vector, payload=payload)]
        )
    except Exception as e:
        print(f"Failed to index chat turn to Qdrant: {str(e)}")


# Chat session management setup
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<h3 style='font-weight: 600; font-size: 1.1rem;'>Chat Sessions</h3>",
    unsafe_allow_html=True,
)

store = get_session_store()
sessions = store.list_sessions()

if "active_session_id" not in st.session_state:
    if sessions:
        st.session_state.active_session_id = sessions[0]["session_id"]
    else:
        new_sid = str(uuid.uuid4())
        store.save_session(new_sid, "New Chat Session", [])
        st.session_state.active_session_id = new_sid
        sessions = store.list_sessions()

# Limit list to top 5 unique recent sessions, ensuring active session is always included
recent_sessions = sessions[:5]
if st.session_state.active_session_id not in [s["session_id"] for s in recent_sessions]:
    active_details = next((s for s in sessions if s["session_id"] == st.session_state.active_session_id), None)
    if active_details:
        recent_sessions.append(active_details)
    else:
        active_details = store.load_session(st.session_state.active_session_id)
        recent_sessions.append({
            "session_id": active_details.get("session_id"),
            "name": active_details.get("name"),
            "created_at": active_details.get("created_at"),
            "updated_at": active_details.get("updated_at")
        })

session_options = {s["session_id"]: s["name"] for s in recent_sessions}

selected_session_id = st.sidebar.selectbox(
    "Select Session",
    options=list(session_options.keys()),
    format_func=lambda x: session_options.get(x, x),
    index=list(session_options.keys()).index(st.session_state.active_session_id) if st.session_state.active_session_id in session_options else 0
)

if selected_session_id != st.session_state.active_session_id:
    st.session_state.active_session_id = selected_session_id
    st.rerun()

if st.sidebar.button("➕ New Chat Session", use_container_width=True):
    new_sid = str(uuid.uuid4())
    store.save_session(new_sid, "New Chat Session", [])
    st.session_state.active_session_id = new_sid
    st.rerun()

active_session = store.load_session(st.session_state.active_session_id)
active_name = active_session.get("name", "New Chat Session")

new_session_name = st.sidebar.text_input("Rename Session", value=active_name)
if new_session_name != active_name and new_session_name.strip():
    store.save_session(st.session_state.active_session_id, new_session_name, active_session.get("messages", []))
    st.rerun()

st.session_state.messages = active_session.get("messages", [])

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input
if user_prompt := st.chat_input(
    "Ask a question about tasks or query the knowledge base..."
):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning..."):
            try:
                # Grab a rolling window of recent history (up to K turns = 2K messages)
                recent_history = st.session_state.messages[:-1][-2 * ROLLING_WINDOW_TURNS:]
                
                final_answer, thoughts = asyncio.run(
                    run_agent_turn(user_prompt, selected_llm, recent_history, st.session_state.active_session_id)
                )

                # Render reasoning steps in expandable list
                if thoughts:
                    with st.expander(
                        "Agent Reasoning & Stdio Tool Call Execution Trace"
                    ):
                        for t_type, t_val in thoughts:
                            if t_type == "thought":
                                st.markdown(f"**Thought:** {t_val}")
                            elif t_type == "observation":
                                st.markdown(f"**Observation:** `{t_val}`")

                st.markdown(final_answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_answer}
                )
                
                # Save session to persistent store
                store.save_session(
                    st.session_state.active_session_id,
                    active_name,
                    st.session_state.messages
                )
                
                # Index the chat turn to Qdrant for long-term semantic memory
                asyncio.run(
                    index_chat_turn_async(
                        st.session_state.active_session_id,
                        user_prompt,
                        final_answer
                    )
                )
                
            except Exception as ex:
                import traceback

                traceback.print_exc()  # Log full traceback to terminal

                error_msg = f"Reasoning Loop Failed: {str(ex)}"
                if hasattr(ex, "exceptions"):
                    sub_errors = []
                    for sub_ex in ex.exceptions:
                        sub_errors.append(f"- {type(sub_ex).__name__}: {str(sub_ex)}")
                    error_msg += "\n\nSub-exceptions details:\n" + "\n".join(sub_errors)
                st.error(error_msg)


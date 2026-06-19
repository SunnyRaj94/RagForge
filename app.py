"""
app.py — RagForge Streamlit UI

Uses the LangGraph-based RagForgeAgent (agents/rag_agent.py) with:
  - MCP tool binding (Qdrant + OpenProject)
  - Human-in-the-loop (HITL) approval before write operations
  - Rolling context window session persistence
  - Qdrant long-term semantic chat history indexing
"""

import asyncio
import json
import os
import uuid

import httpx
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.ragforge.agents.rag import RagForgeAgent, index_chat_turn, HITL_CONFIRM_PROMPT
from src.ragforge.config import (
    OLLAMA_URL,
    OPENPROJECT_API_KEY,
    OPENPROJECT_URL,
    ROLLING_WINDOW_TURNS,
    DEFAULT_LLM_MODEL,
)
from src.ragforge.session_store import get_session_store

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RagForge RAG Agent",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium styling ───────────────────────────────────────────────────────────
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
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(120deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99,102,241,0.15);
        border-color: rgba(99,102,241,0.3);
    }
    .metric-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6366f1;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .thought-container {
        background: rgba(99,102,241,0.08);
        border-left: 3px solid #6366f1;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #c7d2fe;
    }
    .thought-header {
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #818cf8;
        margin-bottom: 0.3rem;
    }
    .observation-container {
        background: rgba(16,185,129,0.08);
        border-left: 3px solid #10b981;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.87rem;
        color: #a7f3d0;
        font-family: 'JetBrains Mono', monospace;
    }
    .observation-header {
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #34d399;
        margin-bottom: 0.3rem;
    }
    .hitl-box {
        background: rgba(245,158,11,0.1);
        border: 1px solid rgba(245,158,11,0.4);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_ollama_models() -> list[str]:
    try:
        res = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        models = res.json().get("models", [])
        return [
            m["name"] for m in models
            if "embed" not in m["name"].lower()
        ]
    except Exception:
        return ["gemma4:e4b"]


def get_openproject_stats() -> tuple[str, int, int]:
    try:
        auth = ("apikey", OPENPROJECT_API_KEY)
        rp = httpx.get(f"{OPENPROJECT_URL}/api/v3/projects", auth=auth, timeout=3.0)
        rwp = httpx.get(f"{OPENPROJECT_URL}/api/v3/work_packages", auth=auth, timeout=3.0)
        return "Connected", rp.json().get("total", 0), rwp.json().get("total", 0)
    except Exception:
        return "Offline", 0, 0


# ── Sidebar — model picker ────────────────────────────────────────────────────
st.sidebar.markdown("<h3 style='font-weight:700;'>⚙️ Model</h3>", unsafe_allow_html=True)
available_models = get_ollama_models()
default_idx = 0
for model_candidate in ["qwen2.5-coder:latest", DEFAULT_LLM_MODEL]:
    if model_candidate in available_models:
        default_idx = available_models.index(model_candidate)
        break
selected_llm = st.sidebar.selectbox("LLM Model", available_models, index=default_idx)

# ── Sidebar — session management ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='font-weight:600;font-size:1.1rem;'>Chat Sessions</h3>", unsafe_allow_html=True)

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

recent_sessions = sessions[:5]
if st.session_state.active_session_id not in [s["session_id"] for s in recent_sessions]:
    active_details = next(
        (s for s in sessions if s["session_id"] == st.session_state.active_session_id), None
    )
    if not active_details:
        active_details = store.load_session(st.session_state.active_session_id)
    recent_sessions.append(active_details)

session_options = {s["session_id"]: s["name"] for s in recent_sessions}

selected_session_id = st.sidebar.selectbox(
    "Select Session",
    options=list(session_options.keys()),
    format_func=lambda x: session_options.get(x, x),
    index=(
        list(session_options.keys()).index(st.session_state.active_session_id)
        if st.session_state.active_session_id in session_options
        else 0
    ),
)

if selected_session_id != st.session_state.active_session_id:
    st.session_state.active_session_id = selected_session_id
    st.session_state.pop("agent", None)   # reset agent on session switch
    st.rerun()

if st.sidebar.button("➕ New Chat Session", use_container_width=True):
    new_sid = str(uuid.uuid4())
    store.save_session(new_sid, "New Chat Session", [])
    st.session_state.active_session_id = new_sid
    st.session_state.pop("agent", None)
    st.rerun()

active_session = store.load_session(st.session_state.active_session_id)
active_name = active_session.get("name", "New Chat Session")

new_session_name = st.sidebar.text_input("Rename Session", value=active_name)
if new_session_name != active_name and new_session_name.strip():
    store.save_session(
        st.session_state.active_session_id,
        new_session_name,
        active_session.get("messages", []),
    )
    st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1 class='main-title'>RagForge RAG Agent</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>LangGraph ReAct Agent · MCP Tools · Human-in-the-Loop · Qdrant Knowledge Base</div>",
    unsafe_allow_html=True,
)

# ── Metrics bar ───────────────────────────────────────────────────────────────
status_msg, p_count, wp_count = get_openproject_stats()
mc1, mc2, mc3 = st.columns(3)
with mc1:
    color = "#10b981" if status_msg == "Connected" else "#ef4444"
    st.markdown(
        f"<div class='metric-card'><div class='metric-header'>OpenProject</div>"
        f"<div class='metric-value' style='color:{color};font-size:1.2rem;'>{status_msg}</div></div>",
        unsafe_allow_html=True,
    )
with mc2:
    st.markdown(
        f"<div class='metric-card'><div class='metric-header'>Active Projects</div>"
        f"<div class='metric-value'>{p_count}</div></div>",
        unsafe_allow_html=True,
    )
with mc3:
    st.markdown(
        f"<div class='metric-card'><div class='metric-header'>Work Packages</div>"
        f"<div class='metric-value'>{wp_count}</div></div>",
        unsafe_allow_html=True,
    )

# ── Agent initialisation (cached per session + model) ────────────────────────

@st.cache_resource(show_spinner="Connecting to MCP servers & building agent graph…")
def get_agent(llm_model: str, session_id: str) -> RagForgeAgent:
    agent = RagForgeAgent(llm_model=llm_model, session_id=session_id)
    asyncio.run(agent.initialise())
    return agent


agent: RagForgeAgent = get_agent(selected_llm, st.session_state.active_session_id)

# Thread config — each session gets its own LangGraph thread for checkpoint isolation
thread_config = {"configurable": {"thread_id": st.session_state.active_session_id}}


def run_async_stream(state_update, thread_config):
    async def _collect():
        events = []
        async for event in agent.graph.astream(
            state_update,
            config=thread_config,
            stream_mode="values",
        ):
            events.append(event)
        return events
    return asyncio.run(_collect())


# ── Restore message history ───────────────────────────────────────────────────
st.session_state.messages = active_session.get("messages", [])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── HITL resume state ─────────────────────────────────────────────────────────
if "hitl_pending" not in st.session_state:
    st.session_state.hitl_pending = False
if "hitl_tool_name" not in st.session_state:
    st.session_state.hitl_tool_name = None
if "hitl_tool_args" not in st.session_state:
    st.session_state.hitl_tool_args = None
if "pending_user_prompt" not in st.session_state:
    st.session_state.pending_user_prompt = None

# ── HITL approval UI (shown when agent paused before a write tool) ─────────
if st.session_state.hitl_pending:
    st.markdown("<div class='hitl-box'>", unsafe_allow_html=True)
    st.warning("🛑 **Human-in-the-Loop** — Agent wants to execute a write operation")
    st.markdown(
        HITL_CONFIRM_PROMPT.format(
            tool_name=st.session_state.hitl_tool_name,
            tool_args=json.dumps(st.session_state.hitl_tool_args, indent=2),
        )
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col_approve, col_reject = st.columns(2)
    with col_approve:
        if st.button("✅ Approve & Execute", use_container_width=True, type="primary"):
            # Resume graph with approval
            with st.spinner("Executing approved tool…"):
                state_update = {"hitl_approved": True, "pending_tool_call": None}
                events = run_async_stream(state_update, thread_config)
                # Grab final answer from last AIMessage without tool calls
                final_answer = ""
                for event in reversed(events):
                    msgs = event.get("messages", [])
                    for m in reversed(msgs):
                        if isinstance(m, AIMessage) and not m.tool_calls and m.content:
                            final_answer = m.content
                            break
                    if final_answer:
                        break

            st.session_state.hitl_pending = False
            if final_answer:
                with st.chat_message("assistant"):
                    st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                store.save_session(
                    st.session_state.active_session_id,
                    active_name,
                    st.session_state.messages,
                )
                asyncio.run(
                    index_chat_turn(
                        st.session_state.active_session_id,
                        st.session_state.pending_user_prompt or "",
                        final_answer,
                    )
                )
            st.rerun()

    with col_reject:
        if st.button("❌ Reject", use_container_width=True):
            state_update = {"hitl_approved": False, "pending_tool_call": None}
            events = run_async_stream(state_update, thread_config)
            final_answer = "Action was rejected by you. Let me know how else I can help."
            for event in reversed(events):
                msgs = event.get("messages", [])
                for m in reversed(msgs):
                    if isinstance(m, AIMessage) and not m.tool_calls and m.content:
                        final_answer = m.content
                        break
                if final_answer:
                    break

            st.session_state.hitl_pending = False
            with st.chat_message("assistant"):
                st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            store.save_session(
                st.session_state.active_session_id,
                active_name,
                st.session_state.messages,
            )
            st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────
if not st.session_state.hitl_pending:
    if user_prompt := st.chat_input("Ask a question or request a task…"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.session_state.pending_user_prompt = user_prompt
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            thought_placeholder = st.empty()
            thoughts: list[tuple[str, str]] = []

            def render_thoughts():
                with thought_placeholder.container():
                    for t_type, t_val in thoughts:
                        css_class = "thought-container" if t_type == "thought" else "observation-container"
                        header = "Thought" if t_type == "thought" else "Observation"
                        header_class = "thought-header" if t_type == "thought" else "observation-header"
                        st.markdown(
                            f"<div class='{css_class}'>"
                            f"<div class='{header_class}'>{header}</div>{t_val}</div>",
                            unsafe_allow_html=True,
                        )

            with st.spinner("Agent is reasoning…"):
                try:
                    # Stream graph execution
                    input_state = {
                        "messages": [HumanMessage(content=user_prompt)],
                        "session_id": st.session_state.active_session_id,
                        "pending_tool_call": None,
                        "hitl_approved": None,
                    }

                    final_answer = ""

                    async def run_and_stream():
                        is_interrupted = False
                        async for event in agent.graph.astream(
                            input_state,
                            config=thread_config,
                            stream_mode="values",
                        ):
                            msgs = event.get("messages", [])
                            for m in msgs:
                                if isinstance(m, AIMessage):
                                    if m.tool_calls:
                                        for tc in m.tool_calls:
                                            thoughts.append(
                                                ("thought", f"Calling tool **`{tc['name']}`** with args:<br><code>{json.dumps(tc['args'], indent=2)}</code>")
                                            )
                                        render_thoughts()
                                    elif m.content:
                                        thoughts.append(("thought", m.content))
                                        render_thoughts()
                                elif isinstance(m, ToolMessage):
                                    thoughts.append(("observation", m.content[:600]))
                                    render_thoughts()

                            # Check if graph is interrupted (waiting for HITL)
                            pending = event.get("pending_tool_call")
                            if pending and event.get("hitl_approved") is None:
                                is_interrupted = True
                                st.session_state.hitl_pending = True
                                st.session_state.hitl_tool_name = pending["name"]
                                st.session_state.hitl_tool_args = pending["args"]
                                break
                        return is_interrupted

                    interrupted = asyncio.run(run_and_stream())

                    if not interrupted:
                        thought_placeholder.empty()
                        # Extract final assistant message
                        snapshot = agent.graph.get_state(thread_config)
                        for m in reversed(snapshot.values.get("messages", [])):
                            if isinstance(m, AIMessage) and not m.tool_calls and m.content:
                                final_answer = m.content
                                break

                        if final_answer:
                            st.markdown(final_answer)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": final_answer}
                            )
                            store.save_session(
                                st.session_state.active_session_id,
                                active_name,
                                st.session_state.messages,
                            )
                            asyncio.run(
                                index_chat_turn(
                                    st.session_state.active_session_id,
                                    user_prompt,
                                    final_answer,
                                )
                            )

                        # Render reasoning trace in expander
                        if thoughts:
                            with st.expander("🔍 Agent Reasoning & Tool Execution Trace"):
                                for t_type, t_val in thoughts:
                                    label = "**Thought:**" if t_type == "thought" else "**Observation:**"
                                    st.markdown(f"{label} {t_val}")
                    else:
                        thought_placeholder.empty()
                        st.rerun()

                except Exception as ex:
                    import traceback
                    traceback.print_exc()
                    st.error(f"Agent error: {str(ex)}")

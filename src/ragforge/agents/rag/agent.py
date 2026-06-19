"""
src/ragforge/agents/rag/agent.py

LangGraph-based ReAct agent with:
  - MCP tool binding (Qdrant + OpenProject stdio servers)
  - Human-in-the-loop (HITL) interrupts on write/mutating tools
  - Streaming support for Streamlit
  - Rolling message window context management
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing_extensions import TypedDict

from src.ragforge.agents.rag.prompts import (
    CHAT_HISTORY_DOCUMENT_TEMPLATE,
    RAG_AGENT_SYSTEM_PROMPT,
)
from src.ragforge.config import (
    CHAT_HISTORY_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    OLLAMA_URL,
    OPENPROJECT_API_KEY,
    OPENPROJECT_URL,
    QDRANT_URL,
    ROLLING_WINDOW_TURNS,
)

# ---------------------------------------------------------------------------
# Write / mutating tool names — these trigger HITL pause
# ---------------------------------------------------------------------------
WRITE_TOOLS = {
    "upsert_documents",
    "ingest_file_or_directory",
    "create_project_task",
    "update_task_status",
    "add_task_comment",
}

# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    pending_tool_call: dict | None  # holds tool call waiting for HITL approval
    hitl_approved: bool | None  # True=approved, False=rejected, None=pending


# ---------------------------------------------------------------------------
# Helper — build LangChain tools from live MCP servers
# ---------------------------------------------------------------------------


def _get_python_bin() -> str:
    app_dir = Path(__file__).resolve().parents[4]
    venv = os.path.join(app_dir, ".venv")
    candidate = os.path.join(venv, "bin", "python")
    return candidate if os.path.isfile(candidate) else "python3"


async def build_mcp_tools() -> tuple[list[BaseTool], dict[str, str]]:
    """
    Connects to Qdrant and OpenProject MCP stdio servers, discovers all tools,
    wraps each into a LangChain BaseTool, and returns:
      - list of tools
      - dict mapping tool_name -> server_label ("qdrant" | "openproject")
    """
    app_dir = str(Path(__file__).resolve().parents[4])
    python_bin = _get_python_bin()

    qdrant_params = StdioServerParameters(
        command=python_bin,
        args=[os.path.join(app_dir, "servers", "qdrant_mcp.py")],
    )
    op_params = StdioServerParameters(
        command=python_bin,
        args=[os.path.join(app_dir, "servers", "openproject_mcp.py")],
    )

    raw_tools: list[dict] = []
    tool_server_map: dict[str, str] = {}

    async with (
        stdio_client(qdrant_params) as (rq, wq),
        stdio_client(op_params) as (rop, wop),
    ):
        async with (
            ClientSession(rq, wq) as sq,
            ClientSession(rop, wop) as sop,
        ):
            await sq.initialize()
            await sop.initialize()

            for t in (await sq.list_tools()).tools:
                raw_tools.append(
                    {
                        "name": t.name,
                        "description": t.description,
                        "schema": t.inputSchema,
                        "server": "qdrant",
                    }
                )
                tool_server_map[t.name] = "qdrant"

            for t in (await sop.list_tools()).tools:
                raw_tools.append(
                    {
                        "name": t.name,
                        "description": t.description,
                        "schema": t.inputSchema,
                        "server": "openproject",
                    }
                )
                tool_server_map[t.name] = "openproject"

    return raw_tools, tool_server_map


# ---------------------------------------------------------------------------
# Tool executor — calls the correct MCP server for a given tool
# ---------------------------------------------------------------------------


async def execute_mcp_tool(tool_name: str, tool_args: dict, server: str) -> str:
    """Execute a named tool on the correct MCP stdio server."""
    app_dir = str(Path(__file__).resolve().parents[4])
    python_bin = _get_python_bin()

    qdrant_params = StdioServerParameters(
        command=python_bin,
        args=[os.path.join(app_dir, "servers", "qdrant_mcp.py")],
    )
    op_params = StdioServerParameters(
        command=python_bin,
        args=[os.path.join(app_dir, "servers", "openproject_mcp.py")],
    )

    if server == "qdrant":
        params = qdrant_params
    else:
        params = op_params

    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, tool_args)
            return result.content[0].text if result.content else "No output."


# ---------------------------------------------------------------------------
# LangGraph graph builder
# ---------------------------------------------------------------------------


class RagForgeAgent:
    """
    Builds and caches a compiled LangGraph graph for a given LLM model.
    Supports human-in-the-loop via interrupt_before=["tools"] pattern
    using MemorySaver checkpointer.
    """

    def __init__(self, llm_model: str, session_id: str):
        self.llm_model = llm_model
        self.session_id = session_id
        self.checkpointer = MemorySaver()
        self.tool_server_map: dict[str, str] = {}
        self.raw_tools: list[dict] = []
        self._graph = None

    async def _discover_tools(self):
        self.raw_tools, self.tool_server_map = await build_mcp_tools()

    def _make_ollama_tools_payload(self) -> list[dict]:
        """Format tools for Ollama /api/chat tools param."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["schema"],
                },
            }
            for t in self.raw_tools
        ]

    def build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph."""
        tool_server_map = self.tool_server_map
        raw_tools = self.raw_tools
        llm_model = self.llm_model
        session_id = self.session_id

        # ── node: agent ──────────────────────────────────────────────────────
        async def agent_node(state: AgentState):
            """Call the LLM. Returns AIMessage (possibly with tool_calls)."""
            messages = state["messages"]
            sid = state.get("session_id", session_id)

            # Build system prompt
            system = SystemMessage(
                content=RAG_AGENT_SYSTEM_PROMPT.format(session_id=sid)
            )

            # Trim to rolling window (keep system + last K turns)
            trimmed = (
                messages[-(2 * ROLLING_WINDOW_TURNS) :]
                if len(messages) > 2 * ROLLING_WINDOW_TURNS
                else messages
            )

            # Call Ollama via httpx (tool_calls format)
            tools_payload = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["schema"],
                    },
                }
                for t in raw_tools
            ]

            lc_messages = [system] + trimmed
            payload_messages = []
            for m in lc_messages:
                if isinstance(m, SystemMessage):
                    payload_messages.append({"role": "system", "content": m.content})
                elif isinstance(m, HumanMessage):
                    payload_messages.append({"role": "user", "content": m.content})
                elif isinstance(m, AIMessage):
                    payload_messages.append(
                        {"role": "assistant", "content": m.content or ""}
                    )
                elif isinstance(m, ToolMessage):
                    payload_messages.append(
                        {"role": "user", "content": f"Observation: {m.content}"}
                    )

            async with httpx.AsyncClient() as client:
                try:
                    resp = await client.post(
                        f"{OLLAMA_URL}/api/chat",
                        json={
                            "model": llm_model,
                            "messages": payload_messages,
                            "tools": tools_payload,
                            "stream": False,
                            "options": {"temperature": 0.0},
                        },
                        timeout=120.0,
                    )
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    print(f"Ollama Request Payload: messages={payload_messages}, tools={tools_payload}")
                    print(f"Ollama Error Response: {resp.text}")
                    raise e
                data = resp.json()

            msg_data = data.get("message", {})
            content = msg_data.get("content", "")
            tool_calls_raw = msg_data.get("tool_calls", [])

            # Build AIMessage
            if tool_calls_raw:
                # Ollama returns tool_calls as list of {function: {name, arguments}}
                lc_tool_calls = []
                for tc in tool_calls_raw:
                    fn = tc.get("function", {})
                    lc_tool_calls.append(
                        {
                            "id": str(uuid.uuid4()),
                            "name": fn.get("name", ""),
                            "args": fn.get("arguments", {}),
                            "type": "tool_call",
                        }
                    )
                ai_msg = AIMessage(content=content, tool_calls=lc_tool_calls)
            else:
                ai_msg = AIMessage(content=content)

            return {"messages": [ai_msg], "pending_tool_call": None}

        # ── node: check_hitl ─────────────────────────────────────────────────
        async def check_hitl_node(state: AgentState):
            """
            For each pending tool call check if it is a write operation.
            If so, mark pending_tool_call so the router pauses for HITL.
            """
            last = state["messages"][-1]
            if not isinstance(last, AIMessage) or not last.tool_calls:
                return {"pending_tool_call": None}

            # Check first tool call for write gate
            tc = last.tool_calls[0]
            if tc["name"] in WRITE_TOOLS:
                return {
                    "pending_tool_call": {
                        "id": tc["id"],
                        "name": tc["name"],
                        "args": tc["args"],
                    },
                    "hitl_approved": None,  # waiting for human
                }

            return {"pending_tool_call": None, "hitl_approved": True}

        # ── node: execute_tools ───────────────────────────────────────────────
        async def execute_tools_node(state: AgentState):
            """Execute all tool calls on the last AIMessage."""
            last = state["messages"][-1]
            if not isinstance(last, AIMessage) or not last.tool_calls:
                return {"messages": []}

            tool_messages = []
            for tc in last.tool_calls:
                name = tc["name"]
                args = tc["args"]
                server = tool_server_map.get(name, "qdrant")
                try:
                    result = await execute_mcp_tool(name, args, server)
                except Exception as e:
                    result = f"Tool error: {str(e)}"

                tool_messages.append(
                    ToolMessage(content=result, tool_call_id=tc["id"], name=name)
                )

            return {
                "messages": tool_messages,
                "pending_tool_call": None,
                "hitl_approved": None,
            }

        # ── node: reject_tool ─────────────────────────────────────────────────
        async def reject_tool_node(state: AgentState):
            """User rejected the tool call — inject a refusal observation."""
            last = state["messages"][-1]
            tool_messages = []
            if isinstance(last, AIMessage) and last.tool_calls:
                for tc in last.tool_calls:
                    tool_messages.append(
                        ToolMessage(
                            content="Tool call rejected by user. Do not retry this action.",
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                    )
            return {
                "messages": tool_messages,
                "pending_tool_call": None,
                "hitl_approved": None,
            }

        # ── router ────────────────────────────────────────────────────────────
        def route_after_agent(state: AgentState) -> Literal["check_hitl", "__end__"]:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "check_hitl"
            return "__end__"

        def route_after_hitl(
            state: AgentState,
        ) -> Literal["execute_tools", "hitl_pause", "reject_tool"]:
            approved = state.get("hitl_approved")
            pending = state.get("pending_tool_call")
            if pending and approved is None:
                return "hitl_pause"  # interrupt here for human input
            if approved is False:
                return "reject_tool"
            return "execute_tools"

        def route_after_tools(state: AgentState) -> Literal["agent", "__end__"]:
            return "agent"

        # ── build graph ───────────────────────────────────────────────────────
        builder = StateGraph(AgentState)
        builder.add_node("agent", agent_node)
        builder.add_node("check_hitl", check_hitl_node)
        builder.add_node("execute_tools", execute_tools_node)
        builder.add_node("reject_tool", reject_tool_node)

        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", route_after_agent)
        builder.add_conditional_edges(
            "check_hitl",
            route_after_hitl,
            {
                "execute_tools": "execute_tools",
                "reject_tool": "reject_tool",
                "hitl_pause": END,
            },
        )
        builder.add_edge("execute_tools", "agent")
        builder.add_edge("reject_tool", "agent")

        # hitl_pause is a virtual node: we interrupt here by raising NodeInterrupt
        # In practice we handle this at app level via interrupt_before
        self._graph = builder.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["execute_tools"],  # pause before any tool execution
        )
        return self._graph

    async def initialise(self):
        """Discover tools and compile the graph."""
        await self._discover_tools()
        self.build_graph()

    @property
    def graph(self):
        return self._graph

    def get_tool_server_map(self) -> dict[str, str]:
        return self.tool_server_map

    def get_write_tools(self) -> set[str]:
        return WRITE_TOOLS

    def get_all_tool_names(self) -> list[str]:
        return [t["name"] for t in self.raw_tools]

    def get_raw_tools(self) -> list[dict]:
        return self.raw_tools


# ---------------------------------------------------------------------------
# Qdrant chat history indexing helper
# ---------------------------------------------------------------------------


async def index_chat_turn(session_id: str, user_msg: str, assistant_msg: str):
    """Embed and upsert a completed chat turn into Qdrant for long-term memory."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from datetime import datetime

    text = CHAT_HISTORY_DOCUMENT_TEMPLATE.format(
        user_message=user_msg, assistant_message=assistant_msg
    )
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": DEFAULT_EMBEDDING_MODEL, "prompt": text},
                timeout=30.0,
            )
            resp.raise_for_status()
            vector = resp.json()["embedding"]

        q = QdrantClient(url=QDRANT_URL)
        if not q.collection_exists(CHAT_HISTORY_COLLECTION):
            q.create_collection(
                collection_name=CHAT_HISTORY_COLLECTION,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

        q.upsert(
            collection_name=CHAT_HISTORY_COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": text,
                        "session_id": session_id,
                        "user_message": user_msg,
                        "assistant_message": assistant_msg,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
            ],
        )
    except Exception as e:
        print(f"[chat history] Failed to index turn: {e}")

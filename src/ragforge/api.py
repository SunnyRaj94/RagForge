import asyncio
import json
import os
import time
import uuid
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)

from src.ragforge.agents.rag.agent import RagForgeAgent
from src.ragforge.config import DEFAULT_LLM_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragforge-api")

app = FastAPI(title="RagForge OpenAI-Compatible Agent API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent: RagForgeAgent = None


@app.on_event("startup")
async def startup_event():
    global agent
    logger.info("Initializing RagForgeAgent and MCP tools...")
    agent = RagForgeAgent(llm_model=DEFAULT_LLM_MODEL, session_id="default")
    await agent.initialise()
    logger.info(f"Agent initialized successfully with model: {DEFAULT_LLM_MODEL}")


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "ragforge-agent",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ragforge",
            }
        ],
    }


async def chat_completion_stream(
    messages: list[dict], thread_id: str
) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": thread_id}}

    # Check if the graph is currently paused before the hitl_pause node
    snapshot = agent.graph.get_state(config)
    is_paused = len(snapshot.next) > 0 and "hitl_pause" in snapshot.next

    last_msg_content = messages[-1]["content"].strip().lower() if messages else ""

    think_open = False

    def open_think():
        nonlocal think_open
        if not think_open:
            think_open = True
            return "<think>\n"
        return ""

    def close_think():
        nonlocal think_open
        if think_open:
            think_open = False
            return "</think>\n"
        return ""

    def make_sse_chunk(content: str) -> str:
        data = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "ragforge-agent",
            "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": None}
            ],
        }
        return f"data: {json.dumps(data)}\n\n"

    try:
        if is_paused:
            logger.info(
                f"Thread {thread_id} is currently paused. Last message content: '{last_msg_content}'"
            )
            if last_msg_content in ["approve", "yes", "y", "ok", "execute"]:
                yield make_sse_chunk(
                    open_think()
                    + "🟢 **Human Approval Received**. Resuming tool execution...\n"
                )
                state_update = {"hitl_approved": True, "pending_tool_call": None}
                agent.graph.update_state(config, state_update, as_node="check_hitl")
                stream = agent.graph.astream(None, config=config, stream_mode="updates")
            elif last_msg_content in ["reject", "no", "n", "cancel"]:
                yield make_sse_chunk(
                    open_think()
                    + "🔴 **Human Rejection Received**. Canceling tool execution...\n"
                )
                state_update = {"hitl_approved": False, "pending_tool_call": None}
                agent.graph.update_state(config, state_update, as_node="check_hitl")
                stream = agent.graph.astream(None, config=config, stream_mode="updates")
            else:
                yield make_sse_chunk(
                    open_think()
                    + "⚠️ **New query received during pause**. Canceling pending tool execution...\n"
                )
                state_update = {"hitl_approved": False, "pending_tool_call": None}
                agent.graph.update_state(config, state_update, as_node="check_hitl")

                # Run to completion to clear state
                async for _ in agent.graph.astream(
                    None, config=config, stream_mode="updates"
                ):
                    pass
                yield make_sse_chunk(close_think() + "Starting new query...\n")

                # Run new query
                user_msg = messages[-1]["content"]
                input_state = {
                    "messages": [HumanMessage(content=user_msg)],
                    "session_id": thread_id,
                    "pending_tool_call": None,
                    "hitl_approved": None,
                }
                stream = agent.graph.astream(
                    input_state, config=config, stream_mode="updates"
                )
        else:
            # Reconstruct history if memory was lost (e.g. server restart)
            snapshot = agent.graph.get_state(config)
            if not snapshot.values:
                logger.info(
                    f"Thread {thread_id} memory is empty. Reconstructing history from payload..."
                )
                history_messages = []
                for m in messages[:-1]:
                    if m["role"] == "user":
                        history_messages.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        history_messages.append(AIMessage(content=m["content"]))
                    elif m["role"] == "system":
                        history_messages.append(SystemMessage(content=m["content"]))

                if history_messages:
                    agent.graph.update_state(config, {"messages": history_messages})

            user_msg = messages[-1]["content"] if messages else ""
            logger.info(f"Starting new query for thread {thread_id}: '{user_msg}'")
            input_state = {
                "messages": [HumanMessage(content=user_msg)],
                "session_id": thread_id,
                "pending_tool_call": None,
                "hitl_approved": None,
            }
            stream = agent.graph.astream(
                input_state, config=config, stream_mode="updates"
            )

        async for update in stream:
            if "agent" in update:
                agent_data = update["agent"]
                msgs = agent_data.get("messages", [])
                for m in msgs:
                    if isinstance(m, AIMessage):
                        if m.tool_calls:
                            yield make_sse_chunk(open_think())
                            for tc in m.tool_calls:
                                args_str = json.dumps(tc["args"], indent=2)
                                is_write = tc["name"] in agent.get_write_tools()
                                if is_write:
                                    # Close thought tag first so user can read approval prompt in main stream
                                    yield make_sse_chunk(
                                        close_think() + f"🛑 **Approval Required**\n"
                                        f"The agent wants to execute a write operation: **`{tc['name']}`** with arguments:\n"
                                        f"```json\n{args_str}\n```\n"
                                        f"Please type **`approve`** to run this action, or **`reject`** to cancel.\n"
                                    )
                                else:
                                    yield make_sse_chunk(
                                        f"⚙️ Calling tool **`{tc['name']}`** with args:\n```json\n{args_str}\n```\n"
                                    )
                        elif m.content:
                            if think_open:
                                yield make_sse_chunk(close_think())
                            yield make_sse_chunk(m.content)

            elif "execute_tools" in update:
                tools_data = update["execute_tools"]
                msgs = tools_data.get("messages", [])
                for m in msgs:
                    if isinstance(m, ToolMessage):
                        yield make_sse_chunk(open_think())
                        content_summary = m.content[:1000] + (
                            "..." if len(m.content) > 1000 else ""
                        )
                        yield make_sse_chunk(
                            f"↳ Tool **`{m.name}`** returned:\n```text\n{content_summary}\n```\n"
                        )

            elif "reject_tool" in update:
                reject_data = update["reject_tool"]
                msgs = reject_data.get("messages", [])
                for m in msgs:
                    if isinstance(m, ToolMessage):
                        yield make_sse_chunk(open_think())
                        yield make_sse_chunk(f"↳ Action canceled: {m.content}\n")

        if think_open:
            yield make_sse_chunk(close_think())

        yield "data: [DONE]\n\n"

    except Exception as ex:
        logger.exception("Error in completion stream")
        yield make_sse_chunk(f"\n❌ **API Gateway Error**: {str(ex)}\n")
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(payload: dict, request: Request):
    messages = payload.get("messages", [])
    stream = payload.get("stream", False)

    # Read conversation ID from LibreChat custom headers
    thread_id = (
        request.headers.get("X-Conversation-Id")
        or payload.get("user")
        or "default-thread"
    )
    logger.info(f"Request received. Stream={stream}, Thread ID={thread_id}")

    if stream:
        return StreamingResponse(
            chat_completion_stream(messages, thread_id), media_type="text/event-stream"
        )
    else:
        full_content = ""
        async for chunk in chat_completion_stream(messages, thread_id):
            if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                try:
                    data = json.loads(chunk[6:].strip())
                    content = data["choices"][0]["delta"].get("content", "")
                    full_content += content
                except Exception:
                    pass

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "ragforge-agent",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

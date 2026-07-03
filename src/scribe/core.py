"""Shared building blocks: MCP session helpers, tool-schema conversion,
and the OpenRouter/OpenAI client. Backend-agnostic — works with whatever
Backend object (obsidian, notion, ...) is passed in."""

import asyncio
import os
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.stdio import stdio_client
from openai import OpenAI

TOOL_CALL_TIMEOUT_SECONDS = 30

WRITE_TOOL_PREFIXES = (
    "create_", "update_", "edit_", "delete_", "move_", "rename_",
    "add_", "remove_", "batch_",
)


def build_llm_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def convert_mcp_tool_to_openai(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        },
    }


def is_write_tool(name: str) -> bool:
    return name.startswith(WRITE_TOOL_PREFIXES)


def split_tools(openai_tools: list) -> tuple[list, list]:
    """Returns (read_only_tools, write_tools)."""
    read_only, write = [], []
    for t in openai_tools:
        (write if is_write_tool(t["function"]["name"]) else read_only).append(t)
    return read_only, write


async def execute_tool_call(tool_name: str, arguments: dict, session) -> str:
    try:
        result = await asyncio.wait_for(
            session.call_tool(tool_name, arguments=arguments),
            timeout=TOOL_CALL_TIMEOUT_SECONDS,
        )
        if result.content and len(result.content) > 0:
            return result.content[0].text
        return "Tool executed but returned no content."
    except asyncio.TimeoutError:
        return (
            f"Error: '{tool_name}' timed out after {TOOL_CALL_TIMEOUT_SECONDS}s. "
            "This operation may be too broad or slow — try a narrower query "
            "(smaller max_results, more specific search terms), or skip it "
            "and continue with the rest of the task."
        )
    except Exception as e:
        return f"Error calling tool: {e}"


async def connect_session(stack: AsyncExitStack, backend, config: dict):
    """Start the backend's MCP subprocess and open a session, kept alive
    for as long as `stack` stays open. Returns (session, openai_tools)."""
    server_params = backend.build_server_params(config)
    devnull = stack.enter_context(open(os.devnull, "w"))
    read_stream, write_stream = await stack.enter_async_context(
        stdio_client(server_params, errlog=devnull)
    )
    session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
    await session.initialize()

    tools_result = await session.list_tools()
    openai_tools = [convert_mcp_tool_to_openai(t) for t in tools_result.tools]
    return session, openai_tools
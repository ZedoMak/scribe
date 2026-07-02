"""Shared building blocks: MCP session helpers, tool-schema conversion,
and the OpenRouter/OpenAI client. Used by the chat loop, one-shot
commands, and doctor."""

import os
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
import asyncio

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the user's Obsidian vault. "
    "Always read a note before updating it so you don't overwrite existing "
    "content by accident, unless the user clearly asks you to replace it. "
    "For tasks that touch many notes (reorganizing, restructuring, bulk "
    "tagging), first list and inspect the relevant notes to build a clear "
    "picture of the vault before making any changes, then work through the "
    "changes methodically one note at a time rather than guessing at "
    "structure up front. If a specific note or operation isn't working "
    "after one or two attempts, don't keep retrying it — skip that note, "
    "note it in your final summary as something you couldn't resolve, and "
    "continue with the rest of the task."
)

TOOL_CALL_TIMEOUT_SECONDS = 30


def build_llm_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def build_server_params(vault_path: str) -> StdioServerParameters:
    return StdioServerParameters(
        command="obsidian-mcp",
        args=[],
        env={
            **os.environ,
            "OBSIDIAN_VAULT_PATH": vault_path,
            "OBSIDIAN_LOG_LEVEL": "ERROR",
        },
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


async def execute_tool_call(tool_name: str, arguments: dict, session) -> str:
    try:
        result = await session.call_tool(tool_name, arguments=arguments)
        if result.content and len(result.content) > 0:
            return result.content[0].text
        return "Tool executed but returned no content."
    except Exception as e:
        return f"Error calling tool: {e}"


async def connect_session(stack: AsyncExitStack, vault_path: str):
    """Start the obsidian-mcp subprocess and open an MCP session, kept
    alive for as long as `stack` stays open. Returns (session, openai_tools).
    Shared by the interactive chat loop, one-shot commands, and doctor,
    so the connection logic — and its errlog fix — lives in exactly one
    place."""
    server_params = build_server_params(vault_path)
    devnull = stack.enter_context(open(os.devnull, "w"))
    read_stream, write_stream = await stack.enter_async_context(
        stdio_client(server_params, errlog=devnull)
    )
    session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
    await session.initialize()

    tools_result = await session.list_tools()
    openai_tools = [convert_mcp_tool_to_openai(t) for t in tools_result.tools]
    return session, openai_tools
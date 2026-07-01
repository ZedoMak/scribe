"""Shared building blocks: MCP session helpers, tool-schema conversion,
and the OpenRouter/OpenAI client. Both the chat loop and the doctor
command build on this."""

import os
from mcp import StdioServerParameters
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the user's Obsidian vault. "
    "Always read a note before updating it so you don't overwrite existing "
    "content by accident, unless the user clearly asks you to replace it."
)


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
            # Quiet the server's own logging; we only want MCP protocol
            # traffic on stdio, not its log lines colliding with our UI.
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
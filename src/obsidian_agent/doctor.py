"""Sanity checks: is obsidian-mcp installed, and can we actually reach
tools on the configured vault. This is test_obsidian_mcp.py, promoted to
a real subcommand."""

import asyncio
import json
import shutil

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from .core import build_server_params


def check_obsidian_mcp_installed() -> bool:
    found = shutil.which("obsidian-mcp") is not None
    if not found:
        print("obsidian-mcp was not found on your PATH.")
        print("Install it first, then re-run 'obsidian-agent doctor'.")
    else:
        print("obsidian-mcp found on PATH.")
    return found


async def check_vault_connection(vault_path: str):
    server_params = build_server_params(vault_path)
    print(f"Connecting to vault: {vault_path}")

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            print("\nAvailable tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description.splitlines()[0]}")

            print("\nListing notes in vault root...")
            result = await session.call_tool("list_notes_tool", arguments={"recursive": True})
            content = result.content[0].text if result.content else "No data"
            try:
                data = json.loads(content)
                items = data.get("items", [])
                print(f"Found {data.get('total', len(items))} notes. First 10:")
                for note in items[:10]:
                    print(f"  - {note.get('path')}")
            except json.JSONDecodeError:
                print("Raw response:", content)


def run_doctor(config: dict):
    if not check_obsidian_mcp_installed():
        raise SystemExit(1)
    asyncio.run(check_vault_connection(config["vault_path"]))
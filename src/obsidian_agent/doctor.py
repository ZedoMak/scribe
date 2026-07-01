"""Sanity checks: is obsidian-mcp installed, and can we actually reach
tools on the configured vault. This is test_obsidian_mcp.py, promoted to
a real subcommand with rich output."""

import asyncio
import json
import os
import shutil

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from . import ui
from .core import build_server_params


def check_obsidian_mcp_installed() -> bool:
    found = shutil.which("obsidian-mcp") is not None
    if not found:
        ui.console.print("[bold red]✗[/] obsidian-mcp was not found on your PATH.")
        ui.console.print("  Install it with: [cyan]pip install obsidian-mcp[/]")
    else:
        ui.console.print("[bold green]✓[/] obsidian-mcp found on PATH.")
    return found


async def check_vault_connection(vault_path: str):
    server_params = build_server_params(vault_path)
    ui.console.print(f"[dim]Connecting to vault:[/] {vault_path}")

    # Same stderr redirect as loop.py — keep obsidian-mcp's own log
    # lines from cluttering the doctor output.
    with open(os.devnull, "w") as devnull:
        async with stdio_client(server_params, errlog=devnull) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                tools_result = await session.list_tools()
                ui.console.print(f"[bold green]✓[/] Connected. {len(tools_result.tools)} tools available:")
                for tool in tools_result.tools:
                    desc = (tool.description or "").strip().splitlines()
                    desc = desc[0] if desc else ""
                    ui.console.print(f"  [cyan]{tool.name}[/]  [dim]{desc}[/]")

                ui.console.print("\n[dim]Listing notes in vault root...[/]")
                result = await session.call_tool("list_notes_tool", arguments={"recursive": True})
                content = result.content[0].text if result.content else "No data"
                try:
                    data = json.loads(content)
                    items = data.get("items", [])
                    ui.console.print(f"[bold green]✓[/] Found {data.get('total', len(items))} notes. First 10:")
                    for note in items[:10]:
                        ui.console.print(f"  [dim]-[/] {note.get('path')}")
                except json.JSONDecodeError:
                    ui.console.print("[yellow]Raw response:[/]", content)


def run_doctor(config: dict):
    ui.console.rule("[bold cyan]obsidian-agent doctor")
    if not check_obsidian_mcp_installed():
        raise SystemExit(1)
    try:
        asyncio.run(check_vault_connection(config["vault_path"]))
    except Exception as e:
        ui.print_error(f"Could not connect to vault: {e}")
        raise SystemExit(1)
    ui.console.print("\n[bold green]All checks passed.[/]")
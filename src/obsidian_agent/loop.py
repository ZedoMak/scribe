"""The agentic chat loop: keeps calling tools until the model responds
with plain text instead of another tool call, instead of stopping after
one round. Now with streaming output and slash commands."""

import json
import os
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from . import ui
from .core import (
    SYSTEM_PROMPT,
    build_llm_client,
    build_server_params,
    convert_mcp_tool_to_openai,
    execute_tool_call,
)

MAX_STEPS = 8
EXTRA_HEADERS = {
    "HTTP-Referer": "http://localhost:8080",
    "X-Title": "Obsidian Agent",
}


async def run_turn(client_llm, model, messages, tools, session, max_steps=MAX_STEPS):
    """Keep calling tools, feeding results back to the model, until it
    responds with plain text instead of another tool call, or the step
    budget runs out."""
    for _ in range(max_steps):
        with ui.thinking_spinner():
            response = client_llm.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7,
                extra_headers=EXTRA_HEADERS,
            )
        msg = response.choices[0].message
        messages.append(msg.to_dict())

        if not msg.tool_calls:
            return msg.content

        for tool_call in msg.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                result_text = "Error: malformed tool arguments"
                args = {}
            else:
                ui.print_tool_call(tool_call.function.name, args)
                result_text = await execute_tool_call(
                    tool_call.function.name, args, session
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                }
            )

    return "Stopped after too many steps — the task may be more complex than expected."


def handle_slash_command(command: str, messages: list, tools: list, system_prompt: str) -> bool:
    """Returns True if the input was a slash command (and was handled)."""
    if command == "/help":
        ui.print_help()
        return True
    if command == "/tools":
        ui.print_tools(tools)
        return True
    if command == "/clear":
        messages.clear()
        messages.append({"role": "system", "content": system_prompt})
        ui.console.print("[dim]Conversation cleared.[/]\n")
        return True
    return False


async def main_loop(config: dict):
    vault_path = config["vault_path"]
    model = config["model"]
    api_key = config["openrouter_api_key"]

    client_llm = build_llm_client(api_key)
    server_params = build_server_params(vault_path)

    # AsyncExitStack keeps the stdio subprocess and MCP session alive for
    # the whole chat loop. Returning a session from inside a narrower
    # "async with" block would close the connection the moment that
    # function returns — every tool call after that would silently fail.
    async with AsyncExitStack() as stack:
        with ui.console.status("[dim]starting obsidian-mcp...[/]"):
            # Redirect the subprocess's stderr to devnull so its own log
            # lines (fastmcp INFO logs, authlib deprecation warnings)
            # don't interleave with our Rich output. A plain sync file
            # object is fine here — errlog just needs to be file-like.
            devnull = stack.enter_context(open(os.devnull, "w"))
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(server_params, errlog=devnull)
            )
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            tools_result = await session.list_tools()

        if not tools_result.tools:
            ui.print_error("No tools available from obsidian-mcp. Exiting.")
            return

        openai_tools = [convert_mcp_tool_to_openai(t) for t in tools_result.tools]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        ui.print_banner(vault_path, model)

        while True:
            try:
                ui.print_turn_separator()
                user_input = ui.prompt_user()
            except (EOFError, KeyboardInterrupt):
                ui.console.print("\n[dim]Goodbye.[/]")
                break

            stripped = user_input.strip()
            if not stripped:
                continue
            if stripped.lower() in ("exit", "quit", "/exit", "/quit"):
                ui.console.print("[dim]Goodbye.[/]")
                break
            if stripped.startswith("/"):
                if handle_slash_command(stripped, messages, openai_tools, SYSTEM_PROMPT):
                    continue

            messages.append({"role": "user", "content": user_input})

            try:
                reply = await run_turn(client_llm, model, messages, openai_tools, session)
                messages.append({"role": "assistant", "content": reply})
                ui.print_agent_reply(reply)
            except Exception as e:
                ui.print_error(str(e))
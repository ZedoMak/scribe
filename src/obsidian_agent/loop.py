"""The agentic chat loop and one-shot task runner. Both share the same
connection setup and tool-calling turn logic — the only difference is
whether we loop on user input afterward (chat) or exit after one task
(one-shot)."""

import json
from contextlib import AsyncExitStack

from . import ui
from .core import (
    SYSTEM_PROMPT,
    build_llm_client,
    connect_session,
    execute_tool_call,
)

# Interactive chat and one-shot tasks get different budgets: a single
# one-shot command like "reorganize my Projects folder" can legitimately
# need many more tool calls than a chat turn answering one question.
MAX_STEPS_CHAT = 10
MAX_STEPS_ONE_SHOT = 40

EXTRA_HEADERS = {
    "HTTP-Referer": "http://localhost:8080",
    "X-Title": "Obsidian Agent",
}


async def run_turn(client_llm, model, messages, tools, session, max_steps):
    """Keep calling tools, feeding results back to the model, until it
    responds with plain text instead of another tool call, or the step
    budget runs out."""
    seen_calls = {}  # signature -> result, so we can catch loops

    for step in range(max_steps):
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
            else:
                signature = (tool_call.function.name, json.dumps(args, sort_keys=True))
                if signature in seen_calls:
                    result_text = (
                        "You already called this exact tool with these exact "
                        "arguments earlier in this task. Re-running it will "
                        "give the same result. Stop retrying it — either use "
                        "the earlier result, try a different approach, or if "
                        "this file/path seems to no longer exist or be "
                        "reachable, skip it and move on to the rest of the task."
                    )
                else:
                    ui.print_tool_call(tool_call.function.name, args)
                    result_text = await execute_tool_call(
                        tool_call.function.name, args, session
                    )
                    seen_calls[signature] = result_text
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                }
            )

    return (
        f"Stopped after {max_steps} steps — this task may need breaking "
        "into smaller pieces, or run it again to continue from here."
    )


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
    """Interactive chat: keeps the vault connection open across many
    turns, and loops on user input until they exit."""
    vault_path = config["vault_path"]
    model = config["model"]
    client_llm = build_llm_client(config["openrouter_api_key"])

    async with AsyncExitStack() as stack:
        with ui.console.status("[dim]starting obsidian-mcp...[/]"):
            session, openai_tools = await connect_session(stack, vault_path)

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
                reply = await run_turn(
                    client_llm, model, messages, openai_tools, session, MAX_STEPS_CHAT
                )
                messages.append({"role": "assistant", "content": reply})
                ui.print_agent_reply(reply)
            except Exception as e:
                ui.print_error(str(e))


async def run_once(config: dict, task_prompt: str):
    """One-shot mode: connect, run a single task to completion (possibly
    many tool calls), print the result, exit. Used by summarize/search/
    tags/task commands."""
    vault_path = config["vault_path"]
    model = config["model"]
    client_llm = build_llm_client(config["openrouter_api_key"])

    async with AsyncExitStack() as stack:
        with ui.console.status("[dim]connecting to vault...[/]"):
            session, openai_tools = await connect_session(stack, vault_path)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt},
        ]

        try:
            reply = await run_turn(
                client_llm, model, messages, openai_tools, session, MAX_STEPS_ONE_SHOT
            )
            ui.print_agent_reply(reply)
        except Exception as e:
            ui.print_error(str(e))
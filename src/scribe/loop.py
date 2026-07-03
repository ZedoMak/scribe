"""The agentic chat loop and one-shot task runner. One-shot tasks that
can mutate the vault go through a plan -> confirm -> execute flow so
destructive actions never happen without the user seeing them first."""

import json
from contextlib import AsyncExitStack

from . import ui
from .core import (
    SYSTEM_PROMPT,
    build_llm_client,
    connect_session,
    execute_tool_call,
    split_tools,
)

MAX_STEPS_CHAT = 10
MAX_STEPS_PLAN = 15
MAX_STEPS_EXECUTE = 40

EXTRA_HEADERS = {
    "HTTP-Referer": "http://localhost:8080",
    "X-Title": "Obsidian Agent",
}

PLAN_INSTRUCTION = (
    "\n\nFirst, investigate the vault using the read-only tools available "
    "to you to understand the current state of the relevant notes and "
    "folders. Do NOT make any changes yet — you don't have write tools "
    "available in this phase. Once you understand what needs to happen, "
    "respond with a clear, numbered, human-readable plan describing "
    "exactly which notes/folders you will create, move, rename, or edit, "
    "and how. Do not call any more tools once you're ready to write the plan."
)

EXECUTE_INSTRUCTION = (
    "The user approved the plan above. Execute it now, exactly as "
    "described, using the tools available to you. Work through it "
    "methodically. If something doesn't match what you expected "
    "(a file already renamed, a folder already existing), don't "
    "improvise a workaround — skip that step, note it, and continue "
    "with the rest of the plan."
)


async def run_turn(client_llm, model, messages, tools, session, max_steps):
    """Keep calling tools, feeding results back to the model, until it
    responds with plain text instead of another tool call, or the step
    budget runs out."""
    seen_calls = {}

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
                        "arguments earlier. Don't retry it — use the earlier "
                        "result, try something different, or skip this and "
                        "move on."
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
    """Interactive chat: full tool access every turn, no plan/confirm
    gate. You're driving in real time, so you see and can react to each
    tool call as it happens rather than approving a batch up front."""
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


async def run_once(config: dict, task_prompt: str, skip_confirm: bool = False):
    """One-shot mode with a plan -> confirm -> execute gate. Phase 1 only
    has read-only tools, so it's physically incapable of changing
    anything while it investigates and proposes a plan. Phase 2 only
    runs if the user approves."""
    vault_path = config["vault_path"]
    model = config["model"]
    client_llm = build_llm_client(config["openrouter_api_key"])

    async with AsyncExitStack() as stack:
        with ui.console.status("[dim]connecting to vault...[/]"):
            session, all_tools = await connect_session(stack, vault_path)

        read_only_tools, write_tools = split_tools(all_tools)

        # If the task has no write tools relevant at all (e.g. summarize,
        # search, tags), there's nothing to confirm — just run it straight
        # through with full tool access.
        if not write_tools:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task_prompt},
            ]
            try:
                reply = await run_turn(
                    client_llm, model, messages, all_tools, session, MAX_STEPS_EXECUTE
                )
                ui.print_agent_reply(reply)
            except Exception as e:
                ui.print_error(str(e))
            return

        # Phase 1: plan, read-only.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt + PLAN_INSTRUCTION},
        ]
        try:
            plan = await run_turn(
                client_llm, model, messages, read_only_tools, session, MAX_STEPS_PLAN
            )
        except Exception as e:
            ui.print_error(str(e))
            return

        ui.print_plan(plan or "(the agent didn't produce a plan)")

        if not skip_confirm:
            if not ui.confirm_plan():
                ui.console.print("[dim]Cancelled. No changes were made.[/]")
                return

        # Phase 2: execute, full tool access.
        messages.append({"role": "assistant", "content": plan})
        messages.append({"role": "user", "content": EXECUTE_INSTRUCTION})
        try:
            result = await run_turn(
                client_llm, model, messages, all_tools, session, MAX_STEPS_EXECUTE
            )
            ui.print_agent_reply(result)
        except Exception as e:
            ui.print_error(str(e))
"""Terminal presentation layer: banner, spinners, markdown rendering,
colored tool-call output. Kept separate from the agent logic so the
core loop doesn't need to know or care how things are displayed."""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text
from rich.rule import Rule

console = Console()

BANNER = r"""
[bold cyan] ██████╗ ██████╗ ███████╗██╗██████╗ ██╗ █████╗ ███╗   ██╗[/]
[bold cyan]██╔═══██╗██╔══██╗██╔════╝██║██╔══██╗██║██╔══██╗████╗  ██║[/]
[bold cyan]██║   ██║██████╔╝███████╗██║██║  ██║██║███████║██╔██╗ ██║[/]
[bold cyan]██║   ██║██╔══██╗╚════██║██║██║  ██║██║██╔══██║██║╚██╗██║[/]
[bold cyan]╚██████╔╝██████╔╝███████║██║██████╔╝██║██║  ██║██║ ╚████║[/]
[bold cyan] ╚═════╝ ╚═════╝ ╚══════╝╚═╝╚═════╝ ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝[/]
[dim]                          a g e n t[/]
"""


def print_banner(vault_path: str, model: str):
    console.print(BANNER)
    console.print(f"[dim]vault:[/] {vault_path}")
    console.print(f"[dim]model:[/] {model}")
    console.print("[dim]Type a message, or /help for commands.[/]\n")


def print_help():
    console.print(
        Panel.fit(
            "[bold]/help[/]   show this message\n"
            "[bold]/tools[/]  list available vault tools\n"
            "[bold]/clear[/]  clear the conversation history\n"
            "[bold]/exit[/]   quit  [dim](also: exit, quit, Ctrl+D)[/]",
            title="commands",
            border_style="cyan",
        )
    )


def print_tools(tools: list):
    lines = []
    for t in tools:
        name = t["function"]["name"]
        desc = (t["function"]["description"] or "").strip().splitlines()
        desc = desc[0] if desc else ""
        lines.append(f"[bold cyan]{name}[/]  [dim]{desc}[/]")
    console.print(Panel.fit("\n".join(lines), title=f"{len(tools)} tools available", border_style="cyan"))


def print_tool_call(name: str, arguments: dict):
    args_str = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    console.print(f"  [dim]→ using[/] [bold yellow]{name}[/]([dim]{args_str}[/])")


def print_agent_reply(text: str):
    console.print()
    console.print("[bold magenta]◆ agent[/]")
    console.print(Markdown(text or "*(no response)*"), style="")
    console.print()


def print_error(message: str):
    console.print(f"[bold red]Error:[/] {message}")


def thinking_spinner():
    """Returns a context-manager Live spinner; use with `with thinking_spinner():`."""
    return Live(Spinner("dots", text=Text(" thinking...", style="dim")), console=console, refresh_per_second=10, transient=True)

def print_turn_separator():
    console.print(Rule(style="dim"))

def prompt_user() -> str:
    return console.input("[bold cyan]❯[/] ")
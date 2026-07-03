"""Command-line entry point for obsidian-agent."""

import asyncio

import typer

from .config import CONFIG_FILE, load_config
from .doctor import run_doctor
from .loop import main_loop, run_once

app = typer.Typer(
    name="obsidian-agent",
    help="Chat with your Obsidian vault via an MCP-connected AI agent.",
    add_completion=True,
    no_args_is_help=False,
)


@app.command()
def chat(
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Interactive mode is the default and only mode for chat. "
        "Use --no-interactive to be redirected to the `task` command instead.",
    )
):
    """Start an interactive chat session."""
    if not interactive:
        typer.echo('`chat` is always interactive. Use: obsidian-agent task "..."')
        raise typer.Exit(1)
    config = load_config()
    asyncio.run(main_loop(config))


@app.command()
def summarize(
    file: str = typer.Option(..., "--file", help="Note path or title to summarize.")
):
    """Summarize a single note, then exit."""
    config = load_config()
    asyncio.run(run_once(config, f"Summarize the note '{file}'."))


@app.command()
def search(
    query: str = typer.Option(..., "--query", help="What to search for.")
):
    """Search the vault for matching notes, then exit."""
    config = load_config()
    asyncio.run(
        run_once(
            config,
            f"Search my vault for notes matching: {query}. "
            "List each matching note with a brief note on why it matched.",
        )
    )


@app.command()
def tags(
    min_usage: int = typer.Option(
        1, "--min-usage", help="Only show tags used at least this many times."
    )
):
    """List tags in the vault, then exit."""
    config = load_config()
    asyncio.run(
        run_once(
            config,
            f"List all tags used in my vault at least {min_usage} time(s), "
            "each with its usage count, sorted by count descending.",
        )
    )


@app.command()
@app.command()
def task(
    prompt: str = typer.Argument(
        ..., help="Any instruction for the agent to carry out, then exit."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the plan confirmation prompt."
    ),
):
    """Run any one-off task against your vault. For tasks that change
    files, shows a plan and asks for confirmation before executing."""
    config = load_config()
    asyncio.run(run_once(config, prompt, skip_confirm=yes))


@app.command()
def doctor():
    """Check that obsidian-mcp is installed and can reach your vault."""
    config = load_config()
    run_doctor(config)


@app.command()
def config(reset: bool = typer.Option(False, "--reset", help="Re-run first-time setup.")):
    """View or reset your configuration."""
    if reset:
        load_config(force_setup=True)
        return
    typer.echo(f"Config file: {CONFIG_FILE}")
    if CONFIG_FILE.exists():
        typer.echo(CONFIG_FILE.read_text())
    else:
        typer.echo("No config yet — run 'obsidian-agent config --reset' to create one.")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        chat()


if __name__ == "__main__":
    app()
"""Command-line entry point for scribe."""

import asyncio

import typer

from .backends import BACKENDS, get_backend
from .config import CONFIG_FILE, load_config
from .doctor import run_doctor
from .loop import main_loop, run_once

app = typer.Typer(
    name="scribe",
    help="A conversational AI agent for your notes, wired up via MCP.",
    add_completion=True,
)


def _make_backend_app(backend_name: str) -> typer.Typer:
    backend_app = typer.Typer(help=f"Commands for the {backend_name} backend.")

    @backend_app.command()
    def chat():
        """Start an interactive chat session."""
        config = load_config(backend_name)
        asyncio.run(main_loop(config, get_backend(backend_name)))

    @backend_app.command()
    def task(
        prompt: str = typer.Argument(..., help="Any instruction for the agent, then exit."),
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip the plan confirmation prompt."),
    ):
        """Run a one-off task. Shows a plan and asks for confirmation
        before making any changes."""
        config = load_config(backend_name)
        asyncio.run(run_once(config, get_backend(backend_name), prompt, skip_confirm=yes))

    @backend_app.command()
    def summarize(file: str = typer.Option(..., "--file", help="Note/page to summarize.")):
        """Summarize a single item, then exit."""
        config = load_config(backend_name)
        asyncio.run(run_once(config, get_backend(backend_name), f"Summarize '{file}'."))

    @backend_app.command()
    def search(query: str = typer.Option(..., "--query", help="What to search for.")):
        """Search, then exit."""
        config = load_config(backend_name)
        asyncio.run(
            run_once(
                config,
                get_backend(backend_name),
                f"Search for items matching: {query}. List each match with "
                "a brief note on why it matched.",
            )
        )

    @backend_app.command()
    def tags(min_usage: int = typer.Option(1, "--min-usage")):
        """List tags, then exit. (Obsidian-style backends only.)"""
        config = load_config(backend_name)
        asyncio.run(
            run_once(
                config,
                get_backend(backend_name),
                f"List all tags used at least {min_usage} time(s), each with "
                "its usage count, sorted by count descending.",
            )
        )

    @backend_app.command()
    def doctor():
        """Check the backend's MCP server is installed and reachable."""
        config = load_config(backend_name)
        run_doctor(config, get_backend(backend_name))

    @backend_app.command(name="config")
    def config_cmd(
        reset: bool = typer.Option(False, "--reset"),
        set_model: str | None = typer.Option(
            None, "--set-model", help="Change the model without redoing full setup."
        ),
    ):
        """View or reset this backend's configuration."""
        if reset:
            load_config(backend_name, force_setup=True)
            return
        if set_model:
            from .config import _read_raw_config, _write_raw_config

            config = _read_raw_config()
            config["model"] = set_model
            _write_raw_config(config)
            typer.echo(f"Model updated to: {set_model}")
            return
        typer.echo(f"Config file: {CONFIG_FILE}")
        if CONFIG_FILE.exists():
            typer.echo(CONFIG_FILE.read_text())
        else:
            typer.echo(f"No config yet — run 'scribe {backend_name} config --reset'.")

    return backend_app

for _name in BACKENDS:
    app.add_typer(_make_backend_app(_name), name=_name)


if __name__ == "__main__":
    app()

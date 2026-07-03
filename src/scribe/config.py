"""User configuration: a one-time interactive setup, stored under the
platform's standard config directory, with env-var overrides for power
users."""

import os
import tomllib
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "obsidian-agent"
CONFIG_FILE = CONFIG_DIR / "config.toml"

DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"


def _write_config(config: dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        for key, value in config.items():
            f.write(f'{key} = "{value}"\n')
    print(f"Saved config to {CONFIG_FILE}")


def run_first_time_setup() -> dict:
    print("First-time setup for obsidian-agent.\n")
    api_key = input("Enter your OpenRouter API key: ").strip()
    vault_path = input("Enter the absolute path to your Obsidian vault: ").strip()
    model = input(f"Model to use [{DEFAULT_MODEL}]: ").strip() or DEFAULT_MODEL

    config = {
        "openrouter_api_key": api_key,
        "vault_path": vault_path,
        "model": model,
    }
    _write_config(config)
    return config


def load_config(force_setup: bool = False) -> dict:
    if force_setup or not CONFIG_FILE.exists():
        config = run_first_time_setup()
    else:
        with open(CONFIG_FILE, "rb") as f:
            config = tomllib.load(f)

    # Env vars always win, for power users / CI / testing.
    config["openrouter_api_key"] = os.environ.get(
        "OPENROUTER_API_KEY", config.get("openrouter_api_key")
    )
    config["vault_path"] = os.environ.get(
        "OBSIDIAN_VAULT_PATH", config.get("vault_path")
    )
    config["model"] = os.environ.get("AGENT_MODEL", config.get("model", DEFAULT_MODEL))

    missing = [k for k in ("openrouter_api_key", "vault_path") if not config.get(k)]
    if missing:
        raise ValueError(
            f"Missing required config: {', '.join(missing)}. "
            f"Run 'obsidian-agent config --reset' to set up again."
        )
    return config
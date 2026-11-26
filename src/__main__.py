"""Entry point of the CLI: ``uv run python -m src <command> [options]``."""

from src.cli import CLI
import fire


def main() -> None:
    """Expose the :class:`~src.cli.CLI` commands through Python Fire."""
    fire.Fire(CLI)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as e:
        print(e)

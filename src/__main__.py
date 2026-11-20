from src.cli import CLI
import fire


def main() -> None:
    fire.Fire(CLI)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as e:
        print(e)

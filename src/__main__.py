import fire
from src.index import build_repository_index


class CLI:
    def index(self, max_chars=2000):
        build_repository_index(max_chars)



def main():
    fire.Fire(CLI)


if __name__ == "__main__":
    main()


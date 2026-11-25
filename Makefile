PYTHON  := uv run python
SRC     := src
ARGS    ?= index

.DEFAULT_GOAL := help
.PHONY: help install run debug lint lint-strict clean fclean

help:
	@echo "Targets:"
	@echo "  install      sync dependencies with uv"
	@echo "  run          run the app (override with ARGS=..., e.g. make run ARGS='search')"
	@echo "  debug        run the app under pdb"
	@echo "  lint         flake8 + mypy"
	@echo "  lint-strict  flake8 + mypy --strict"
	@echo "  clean        remove caches and bytecode"
	@echo "  fclean       clean + remove the virtualenv"

install:
	uv sync

run: install
	$(PYTHON) -m $(SRC) $(ARGS)

debug: install
	$(PYTHON) -m pdb -m $(SRC) $(ARGS)

lint: install
	uv run flake8 $(SRC)
	uv run mypy $(SRC) \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict: install
	uv run flake8 $(SRC)
	uv run mypy $(SRC) --strict

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -not -path "./.venv/*" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete

fclean: clean
	rm -rf .venv
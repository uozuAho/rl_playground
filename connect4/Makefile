init:
	uv sync

main:
	uv run src/main.py

test:
	uv run pytest

pc:
	uv run ruff format
	uv run ruff check --fix
	uv run pytest

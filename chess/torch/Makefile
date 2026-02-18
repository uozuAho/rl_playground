init:
	uv sync

pc:
	uv run ruff format
	uv run mypy .
	uv run ruff check --fix
	uv run pytest

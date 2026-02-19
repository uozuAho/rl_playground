init:
	uv sync

pc:
	uv run ruff format
	uv run ruff check --fix
	uv run ty check
	uv run pytest

rank:
	uv run src/rank_bots.py

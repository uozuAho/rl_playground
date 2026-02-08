.PHONY: init pc clean-models clean-py clean-full test

clean-py:
	rm -rf .venv
	uv sync

clean-models:
	rm -rf trained_models

clean-full: clean-py clean-models

init:
	uv sync

test-fast:
	uv run pytest -m "not slow"

# precommit
pc:
	uv run ruff format
	uv run ruff check --fix
	uv run pytest

# precommit fast
pcf:
	uv run ruff format
	uv run ruff check --fix
	uv run pytest -m "not slow"

trainaz:
	uv run src/train_az.py

prof-trainaz:
	uv run python -m cProfile src/train_az.py profile > profile.txt

rank:
	uv run src/rank_bots.py

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

train:
	uv run src/train_az.py

profile:
	uv run python -m cProfile src/train_az.py profile > profile.txt

claude:
	./claude_follow_instructions.sh

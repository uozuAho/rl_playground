init:
	uv sync

test:
	uv run pytest

pc:
	uv run ruff format
	uv run ruff check --fix
	uv run pytest

train:
	uv run src/train_az.py

rank:
	uv run src/rank_bots.py

trainmp:
	uv run src/train_az_mp.py

profilemp:
	uv run python -m cProfile src/train_az_mp.py profile_player > profile_mp.txt

profile:
	uv run python -m cProfile src/train_az.py profile > profile.txt

claude:
	./claude_follow_instructions.sh

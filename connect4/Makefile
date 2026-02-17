init:
	uv sync

test:
	uv run pytest

# precommit: format, lint, test
pc:
	uv run ruff format
	uv run ruff check --fix
	uv run pytest

# single threaded training. Slower but maybe easier to understand.
train:
	uv run src/train_az.py

# play bots against each other to determine strength
rank:
	uv run src/rank_bots.py

# multiprocess training. Gets the most out of your CPU/GPU. May need some manual tweaking
# to get best perf.
trainmp:
	uv run src/train_az_mp.py

profilemp:
	uv run python -m cProfile src/train_az_mp.py profile_player > profile_mp.txt

profile:
	uv run python -m cProfile src/train_az.py profile > profile.txt

claude:
	./claude_follow_instructions.sh

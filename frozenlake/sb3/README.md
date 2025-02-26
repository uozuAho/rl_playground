My attempt at frozen lake DQN (?). Not sure if it's actually DQN or some other
'deep' learning. Anyway the goal is to use stable baselines on a discrete
space.

Todo
- may be worth trying optuna to find better learning params, as a model trained
  with 100k iterations is still pretty crap

# quick start
```sh
uv sync
uv run ppo.py
```

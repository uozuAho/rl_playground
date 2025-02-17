# Tic tac toe

Various agents. Some demonstrate the usage of 'invalid action masking':
https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

# quick start
Install uv

```sh
uv sync
uv run pytest
uv rur tabular.py  # tabular agents
uv run dqn.py      # dqn agent. Doesn't do as well as tabular vs random, doesn't
                   # train properly against perfect - always loses
uv run ppo.py      # ppo agent .. doesn't learn properly, avg return -1 (always loses)
uv run mask_ppo.py # maskable ppo agent (don't do invalid actions)
                   # doesn't do any better than a random agent
```

# todo
- optimise training: aim for highest average return with lowest training
    - optuna dqn
    - optuna tabular?
    - interesting variables i can think of:
        - net arch, opponent, num episodes
- inline todos

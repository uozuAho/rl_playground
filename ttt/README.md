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
- try to get better performance from deep model(s)
    - note: maskppo does no better than random
    - note: ppo stays at avg -1 return during training, always does invalid actions
- inline todos

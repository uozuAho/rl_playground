# Tic tac toe

Various agents. Some demonstrate the usage of 'invalid action masking':
https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

# quick start
Install uv

```sh
uv sync
uv run pytest
uv run main.py   # sb3 agents
uv rur opt_tabular.py  # tabular agents
```

# todo
- rename this folder
- try to get better performance from deep model(s)
    - note: ppo does no better than random
    - try dqn?
        - no mask available. Use as ref: https://github.com/google-deepmind/open_spiel/blob/d99705de2cca7075e12fbbd76443fcc123249d6f/open_spiel/python/examples/tic_tac_toe_dqn_vs_tabular.py
- inline todos
- (maybe) allow invalid actions (doesn't update board), compare training
  performance with non-masked

# Tic tac toe

Various agents. Some demonstrate the usage of 'invalid action masking':
https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

# quick start
Install uv

```sh
uv sync
uv run pytest
uv run ppo.py      # ppo agent
uv run mask_ppo.py # maskable ppo agent (don't do invalid actions)
uv rur tabular.py  # tabular agents
```

# todo
- try to get better performance from deep model(s)
    - note: maskppo does no better than random
    - note: ppo stays at avg -1 return during training, always does invalid actions
    - try dqn
        - example: dqn/policy gradient with pytorch: https://github.com/kaifishr/TicTacToe/tree/main
            - idea from here: env returns negative reward + game over for illegal actions
- inline todos
- (maybe) allow invalid actions (doesn't update board), compare training
  performance with non-masked

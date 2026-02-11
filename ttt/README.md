# Tic tac toe

Various agents. Some demonstrate the usage of 'invalid action masking':
https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

Most train as x, and don't play well as o.

# quick start
Install uv

```sh
make init
make pc
make rank
# see makefile for more

uv run sb3-dqn-tuna.py # use optuna to find good hyperparams for DQN
uv run optuna-dashboard sqlite:///dqn-ttt.db  # see hyperparam report
uv run tabular-param-search.py  # try a range of parameters for training. Not
                                # as smart as optuna
```

# todo
- WIP: az mp
    - plot stats from log
        - steps vs pol val loss
    - compare to old train_az. learns faster?
    - maybe: support multiple epochs on replay buffer
    - maybe: auto-balance step gen to learner throughput
    - maybe: save model snapshots
- see my other todo notes
- add `ty check` to make pc. lots of errors

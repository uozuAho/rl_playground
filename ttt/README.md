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
- WIP: perf: try to max out GPU during az mp.
    - WIP log to file and cli
    - don't print stacks on keyboard interrupt
    - ensure close to 100% GPU utilisation
        - put trajectories on step queue instead of steps?
    - add evaluator
    - write metrics to log?
    - update docs: mp is overkill for ttt, just a POC for larger games, see non mp version
- see my other todo notes
- add `ty check` to make pc. lots of errors

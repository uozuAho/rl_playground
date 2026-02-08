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
- WIP experiment: max out gpu in train and eval
    - NOTE threading made things slower, probably due to GIL
    - idea: multiprocessing. start with 3 procs: game step gen, tensor batching, net updater
- perf: maybe: follow answer.md
- see my other todo notes
- add `ty check` to make pc. lots of errors

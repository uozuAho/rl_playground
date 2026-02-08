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
- alpha zero
    - later maybe: experiments:
        - optimise. microoptimisations plus ideas below. eg. vectorise stuff with np,
          batch NN passes where possible
        - encode current player in model input, rather than manual twiddling
        - maybe: discount trajectory values. eg first move of a winning trajectory should be < 1
        - don't update model if evaluation is worse than previous model
        - compress net weight types etc.? eg. don't need float32 to represent TTT board cells
        - do i need to tune learning rate to batch size, epochs etc?
            - maybe. Prob fine for now?
- add `ty check` to make pc. lots of errors

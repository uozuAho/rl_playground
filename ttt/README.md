# Tic tac toe

Various agents. Some demonstrate the usage of 'invalid action masking':
https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

Most train as x, and don't play well as o.

# quick start
Install uv

```sh
make init
make pc
uv run src/rank_bots.py
uv run src/train_bots/train_az.py

uv run sb3-dqn-tuna.py # use optuna to find good hyperparams for DQN
uv run optuna-dashboard sqlite:///dqn-ttt.db  # see hyperparam report
uv run tabular-param-search.py  # try a range of parameters for training. Not
                                # as smart as optuna
```

# todo
- alpha zero
    - WIP compare perf with azfs
        - azfs pretrained, net 4 64, mcts 10:
            as X vs rng, WLD:  98 0 2
            as O vs rng, WLD:  91 1 8
            as X vs perfect, WLD:  0 0 100
            as O vs perfect, WLD:  0 0 100
        - same-ish strength after 5000 training games, mcts 60 train, 10 eval
    - maybe: why az nn tab still lose with good tab accuracy?
        - worsening max loss while improving avg loss. chatty suggestions:
            - WARNING I think this all may be a wild goose chase. AZ train gets high
              win rates before too long, don't worry about fitting an NN to a value
              table unless you really care
            - TLDR
                - WIP adjust loss
                    - DONE try cross entropy instead of kldiv on policy
                    - WIP maybe add small tail penalty: mean_loss + 0.1 * mean(top 5% errors)
                        - todo: seems to work.
                            - does it make sense?
                            - play with coefficient + weight  decay
                            - compare training metrics with earlier. add necessary detail to plot
                - increase weight decay (penalty for large weights)
            - ask claude for help
            - add a term that focuses on tail
                - loss = mean(error) + C*max(error) OR C*mean(top k errors)
            - curriculum learning
            - regularization
                - stronger weight decay
                - gradient penalties
                - lipschitz constraints
        - WIP try bigger/different net arches, learning rates, more?
        - maybe: training metrics: check early vs later game pv loss
        - DONE debug trained net vs random. something's gotta be wrong
            - DONE puct seems correct but weird. can get 18 visits to one node while others
                   have zero visits. AZ tab does same, this also seems to be how ospiel does it
    - why az_tab30 much better than perfect?
    - maybe: add temperature. seems not imporant - in os, T=1, in azfs, T=1.25 during training
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

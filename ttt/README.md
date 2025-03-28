# Tic tac toe

Various agents. Some demonstrate the usage of 'invalid action masking':
https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

# quick start
Install uv

```sh
uv sync
./precommit.sh
uv rur tabular.py  # tabular agents
uv run sb3-ppo.py      # ppo agent .. doesn't learn properly, avg return -1 (always loses)
uv run sb3-mask_ppo.py # maskable ppo agent (don't do invalid actions)
                   # doesn't do any better than a random agent

uv run bot-showdown.py  # all vs all bot fight

uv run sb3-dqn-tuna.py # find good hyperparams
uv run optuna-dashboard sqlite:///dqn-ttt.db  # see hyperparam report
```

# parameter tuning study results
## DQN
- most important params, 50k training steps, random opponent
    - learning rate: best ~0.0001, worst ~0, >0.1
    - final epsilon: best 0.05-0.1, worst ~0, >0.15
    - target update interval: best 300-2200, worst > 4k
    - learning starts: best 1500-4000, worst > 4k
    - (sort of) batch size: higher = slightly better. best 150-250
- top 3 param sets, all got 1.0 eval score:
    - 1 - gets to >90% win rate vs random after about 6k-10k episodes
        - gamma 0.9848576847592656
        - net_arch 32_32
        - act_fn relu
        - batch_size 216
        - buffer_size 4384
        - learning_rate 0.0028750566941175937
        - learning_starts 2826
        - tgt_update_int 392
        - eps_init 0.8135260307997546
        - eps_final 0.08186586832492276
    - 2
        - gamma 0.9930957050483905
        - net_arch 32_32
        - act_fn relu
        - batch_size 164
        - buffer_size 1514
        - learning_rate 0.0011934954427902825
        - learning_starts 2798
        - tgt_update_int 2215
        - eps_init 0.612134701698312
        - eps_final 0.05294021338493515
    - 3
        - gamma 0.9982097827842092
        - net_arch 32_32
        - act_fn relu
        - batch_size 158
        - buffer_size 1978
        - learning_rate 0.0006142379744249534
        - learning_starts 1851
        - tgt_update_int 1627
        - eps_init 0.6919016329828642
        - eps_final 0.07907417425388644

# Chess

# Quick start
Install uv + make.

```sh
make init
make pc
# see makefile for more
```

# todo
- check pychess perf. how many games/steps/sec vs rando?
- make chessagent a protocol instead of ABC
- find opponents of various strength. fast to run?
- if py perf ok: copy paste azmp. does it train fast enough?
    - start with a small network. only net big net for max perf

# maybe/later
- old: greedy_agent (value net)
    - make & train small net
    - evaluate vs opponents
        - random
        - greedy
        - andoma
        - (maybe) mcts with random rollout

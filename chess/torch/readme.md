# Chess

# Quick start
Install uv + make.

```sh
make init
make pc
# see makefile for more
```

# todo
- try ty instead of mypy

NEW
- check pychess perf. how many games/steps/sec vs rando?
- find opponents of various strength. fast to run?
- if py perf ok: copy paste azmp. does it train fast enough?
    - start with a small network. only net big net for max perf

- old: greedy_agent (value net)
    - make & train small net
    - evaluate vs opponents
        - random
        - greedy
        - andoma
        - (maybe) mcts with random rollout

# Chess

# Quick start
Install uv + make.

```sh
make init
make pc
# see makefile for more
```

# todo
- WIP implement ranker
    - rm player from random agent
- find opponents of various strength
    - rank em: game strength and move time/perf
- if py perf ok: copy paste azmp. does it train fast enough?
    - start with a small network. only net big net for max perf
    - see inline todos: add capture reward?

# maybe/later
- alphazero: use full move representation (not just 64x64 from-to squares)
    - read the original paper or ask chatty
- old: greedy_agent (value net)
    - make & train small net
    - evaluate vs opponents
        - random
        - greedy
        - andoma
        - (maybe) mcts with random rollout

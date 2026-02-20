# Chess

# Quick start
Install uv + make.

```sh
make init
make pc
# see makefile for more
```

# todo
- WIP fix azmp code
    - WIP changing from ndarray prob dist to dict{move:prob}
- azmp: does it train fast enough?
    - start with a small network. only net big net for max perf
    - see inline todos: add capture reward?

# maybe/later
- azmp perf
    - use/store move uci instead of move classes
- maybe find more opponents
    - maybe try mcts andoma
- alphazero:
    - use full move representation (not just 64x64 from-to squares)
        - read the original paper or ask chatty
    - copy original alphazero input shape
        - eg. 8 historical positions, knights etc.
- old: greedy_agent (value net)
    - make & train small net
    - evaluate vs opponents
        - random
        - greedy
        - andoma
        - (maybe) mcts with random rollout

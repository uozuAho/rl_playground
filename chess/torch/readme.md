# Chess

# Quick start
Install uv + make.

```sh
make init
make pc
# see makefile for more
```

# todo
- azmp player perf
    - profile
    - maybe use/store move uci instead of move classes
- try on big machine
    - first:
        - run on small remote machine to iron out teething issues
        - print perf report from log
        - copy log from big machine
        - plot perf metrics
- azmp: does it learn?

# maybe/later
- see inline todos: add capture reward?
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

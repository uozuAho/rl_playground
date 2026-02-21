# Chess

It's slow. Even using multiprocessing to max out my whole computer, I'm only
getting < 1 game per second. Most recent effort and todos are around azmp:
multiprocess alphazero. Dunno if it works or not, since training is so slow
it's going to take weeks to see if there's any improvement.

# Quick start
Install uv + make.

```sh
make init
make pc
# see makefile for more
```

# todo
- port to C#. py chess players are very slow, gonna be hard to speed up
- azmp player perf
    - profile: heavy: copy and is_terminal
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

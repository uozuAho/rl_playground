Attempting this: https://github.com/arjangroen/RLC , but redoing everything
with uv & torch, to keep track of dependencies and not have to use tensorflow.

# todo
- WIP: train linear network, simple
    - todo:
        - limit episode length (doesn't stop at 25)
        - train longer. does loss/reward change?
    - simple reqs:
        - no replay memory: compare cpu/gpu perf
        - no gradient clipping
        - no double learning
        - no epsilon schedule
- inline todos
- add stuff under simple reqs
- try conv network
- (maybe) try reinforce alg

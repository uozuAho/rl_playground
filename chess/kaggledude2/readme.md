Attempting this: https://github.com/arjangroen/RLC , but redoing everything
with uv & torch, to keep track of dependencies and not have to use tensorflow.

# todo
- add:
    - replay memory: compare cpu/gpu perf
    - gradient clipping
    - double learning
    - epsilon schedule
- does loss/avg reward change with training?
    - print avgs during training
    - notes
        - simple training, linear model: loss appears to slowly increase after
          5000 eps. no noticeable change in reward.
- inline todos
- add stuff under simple reqs
- try conv network
- (maybe) try reinforce alg

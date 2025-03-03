Attempting this: https://github.com/arjangroen/RLC , but redoing everything
with uv & torch, to keep track of dependencies and not have to use tensorflow.

# todo
- WIP: add:
    - DONE: replay memory
    - gradient clipping
    - double learning
    - epsilon schedule
- does loss/avg reward change with training?
    - if not, probably broken. debug. check common torch / RL mistakes
    - print avgs during training
    - notes
        - simple training, linear model: loss appears to slowly increase after
          5000 eps. no noticeable change in reward.
- inline todos
- try conv network
- training perf:
    - any common mistakes I'm making?
    - compare cpu/gpu
    - float32/16 instead of 64?
- (maybe) try reinforce alg

Attempting this: https://github.com/arjangroen/RLC , but redoing everything
with uv & torch, to keep track of dependencies and not have to use tensorflow.

# todo
- does loss/avg reward change with training?
    - what is the kaggledude plotting in https://www.kaggle.com/code/arjanso/reinforcement-learning-chess-3-q-networks
    - watch a game played by the trained agent vs random
    - try low gamma. kaggle dude uses 0.1
    - check common torch / RL mistakes
    - add periodic loss/reward/eval metrics during long training
    - notes
        - conv net training, 5000eps:
            - loss creeps up, peaks at 800 at about 800 eps
            - reward starts growing after this peak, seems to hit a max of ~40
              after about 1000 eps (39 is top score without any pawn promotions)
            - loss & reward drop at about 3600 eps ... why?
        - simple training, linear model: loss appears to slowly increase after
          5000 eps. no noticeable change in reward.
- training perf:
    - any common mistakes I'm making?
    - compare cpu/gpu
    - float32/16 instead of 64?
- (maybe) try reinforce/policy gradient alg

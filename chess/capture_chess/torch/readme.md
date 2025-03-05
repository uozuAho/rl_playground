Attempting this: https://github.com/arjangroen/RLC , but redoing everything
with uv & torch, to keep track of dependencies and not have to use tensorflow.

I couldn't get the original code working as I couldn't find a compatible
combination of python + tensorflow + other dependencies.

# quick start
```sh
uv sync
uv run simple_conv.py  # read the script, alter params as u like
```

# todo
- train with sb3
    - try optuna
- do [mcts](https://www.kaggle.com/code/arjanso/reinforcement-learning-chess-5-tree-search)
    - full chess
    - bigger net
    - tree search for planning
- maybe
    - do [policy gradients](https://www.kaggle.com/code/arjanso/reinforcement-learning-chess-4-policy-gradients)
    - save trained model
    - watch a game played by the trained/untrained agent
    - try low gamma. kaggle dude uses 0.1
    - add periodic loss/reward/eval printouts during long training
    - sanity check: ensure train/eval reward make sense
        - eval should be a bit higher, use model.eval()
    - notes
        - conv net training, 5000eps:
            - loss creeps up, peaks at 800 at about 800 eps
            - reward starts growing after this peak, seems to hit a max of ~40
              after about 1000 eps (39 is top score without any pawn promotions)
            - loss & reward drop at about 3600 eps ... why?
        - simple training, linear model: loss appears to slowly increase after
          5000 eps. no noticeable change in reward.
- (maybe) training perf:
    - https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    - compare cpu/gpu
    - float32/16 instead of 64?
- (maybe) try reinforce/policy gradient alg

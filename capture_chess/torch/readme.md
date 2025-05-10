# Capture chess

Attempting this: https://github.com/arjangroen/RLC , but redoing everything
with uv & torch, to keep track of dependencies and not have to use tensorflow.

Kaggle pages
- q learning: https://www.kaggle.com/arjanso/reinforcement-learning-chess-3-q-networks
- policy gradients: https://www.kaggle.com/code/arjanso/reinforcement-learning-chess-4-policy-gradients

I couldn't get the original code working as I couldn't find a compatible
combination of python + tensorflow + other dependencies.

# quick start
```sh
uv sync
uv run envtest.py      # check the chess env works as expected
uv run linear_net.py   # q-learning with FC NN, trains slowly, doesn't do well
uv run simple_conv.py  # read the script, alter params as u like
./precommit.sh         # lint, format etc
```

# notes
- conv net training, 5000eps:
    - loss creeps up, peaks at 800 at about 800 eps
    - reward starts growing after this peak, seems to hit a max of ~40
      after about 1000 eps (39 is top score without any pawn promotions)
    - loss & reward drop at about 3600 eps ... why?
- simple training, linear model: loss appears to slowly increase after 5000 eps.
  no noticeable change in reward.


# todo
- WIP: split into file per method, eg linear, conv, policy gradients etc.
    - DONE: linear
    - conv
    - move get_nn_move to chessnet?
- print net details
- print time taken by training
- eval model after training
- https://tqdm.github.io/ progress bar?
- add simple_conv details to readme
    - add training time estimate to readme
- add more params to simple_conv script, print config & details on start
- maybe
    - can i get ok performance?
    - add periodic eval to train?
    - does sb3 do better?
    - why does envtest sometimes fail with > 39 reward?

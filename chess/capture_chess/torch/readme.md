# Capture chess

Attempting this: https://github.com/arjangroen/RLC , but redoing everything
with uv & torch, to keep track of dependencies and not have to use tensorflow.

I couldn't get the original code working as I couldn't find a compatible
combination of python + tensorflow + other dependencies.

# quick start
```sh
uv sync
uv run simple_conv.py  # read the script, alter params as u like
```

# notes
- conv net training, 5000eps:
    - loss creeps up, peaks at 800 at about 800 eps
    - reward starts growing after this peak, seems to hit a max of ~40
        after about 1000 eps (39 is top score without any pawn promotions)
    - loss & reward drop at about 3600 eps ... why?
- simple training, linear model: loss appears to slowly increase after 5000 eps.
  no noticeable change in reward.

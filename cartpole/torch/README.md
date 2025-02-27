# Cartpole

env: https://gymnasium.farama.org/environments/classic_control/cart_pole/

2 actions: left/right
Aim is to keep the pole upright, and the cart away from the edges of the screen.

# Quick start
```sh
uv sync
uv run reinforce.py
```

# DQN todo
- add evaluations before/after training
- add visualisations before/after training
- make simpler versions?
    - no batching
    - no double learning
    - no gradient clipping?

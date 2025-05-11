# Capture chess

Training neural net agents on 'capture chess'. Built referencing:

- q learning: https://www.kaggle.com/arjanso/reinforcement-learning-chess-3-q-networks
- policy gradients: https://www.kaggle.com/code/arjanso/reinforcement-learning-chess-4-policy-gradients
- code: https://github.com/arjangroen/RLC

Redoing everything with uv & torch, to keep track of dependencies and not have
to use tensorflow.

I couldn't get the original code working as I couldn't find a compatible
combination of python + tensorflow + other dependencies.

Capture chess rules:
- max 25 moves
- agent plays white
- opponent is part of the environment and makes random moves
    - see https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/capture_chess/environment.py#L85
- rewards for capturing pieces
    - no negative reward for losing pieces
- reward at end of episode = 0

# quick start
```sh
uv sync
uv run pytest
uv run linear_qlearn.py  # q-learning with FC NN, trains slowly, doesn't do well
uv run conv_qlearn.py    # q-learning with convolutional net. Does better than linear.
                         # Trains from 1000 episodes in ~90 seconds.
./precommit.sh           # lint, format etc
```

# notes
- conv net training, 5000eps:
    - loss creeps up, peaks at 800 at about 800 eps
    - reward starts growing after this peak, seems to hit a max of ~40 after
      about 1000 eps (39 is top score without any pawn promotions or capturing
      promoted pieces)
    - loss & reward drop at about 3600 eps ... why?
- simple training, linear model: loss appears to slowly increase after 5000 eps.
  no noticeable change in reward.
## My summary of RLC code
- [RLC](https://github.com/arjangroen/RLC)
```py
board = Board()
agent = Agent(network='conv',gamma=0.1,lr=0.07)
R = Q_learning(agent,board)
R.agent.fix_model()   # saves the network state to separate storage
pgn = R.learn(iters=750)
```
- [learn](https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/capture_chess/learn.py#L26)
    - play_game N games, N = iters
    - fix_model each 10 games: fixed_model is the target network
        - this is double learning
- [play_game](https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/capture_chess/learn.py#L53)
    - play N games
    - each move
        - save to replay memory: [state, (move from, move to), reward, next_state]
        - remove old moves if memory is full
        - add 1 to sampling_probs   # todo: what's this?
        - update_agent
- [update_agent](https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/capture_chess/learn.py#L139C9-L139C21)
    - sample replay memory
    - self.agent.network_udpate: update model with sample
    - update sampling_probs with returned td errors  # todo: why?
- [network_udpate](https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/capture_chess/agent.py#L111)
    - update the model


# todo
- consolidate linear & conv scripts
- plot periodic eval scores
- add more params to conv script, print config & details on start
- train faster: cpu/gpu/other perf tricks?
- can i get ok agent performance
    - does fc do ok at all?
- maybe
    - https://tqdm.github.io/ progress bar to training and eval
    - perf: chess is slow. multiproc?
    - does sb3 do better?

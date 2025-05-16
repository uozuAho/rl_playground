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
uv run qlearn.py    # test various q-learning agents. tweak params, run
uv run reinforce.py # train an agent using reinforce (broken?)
./precommit.sh      # lint, format etc
```

# notes
## RLC part 3 - q learning
- [q learning](https://www.kaggle.com/arjanso/reinforcement-learning-chess-3-q-networks)
- `uv run qlearn.py`
- trains a state-action value network. Network output is action values for a
  given state
## RLC part 4 - policy gradients
- [policy gradients](https://www.kaggle.com/code/arjanso/reinforcement-learning-chess-4-policy-gradients)
- policy gradient methods learn a policy directly, rather than estimating state
  action values
## My summary of RLC code
- [RLC](https://github.com/arjangroen/RLC)
```py
board = Board()
agent = Agent(network='conv',gamma=0.1,lr=0.07)
R = Q_learning(agent,board)
R.agent.fix_model()   # saves the network state to separate storage
pgn = R.learn(iters=750)
```
### Q learning
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
### Reinforce
- [learn](https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/capture_chess/learn.py#L170)
    - play n games, update the model after each game
        - this differs to q-learning which updates the model after every move
    - each game
        - predict actions with the current model
    - at the end of the game: update model = reinforce_agent = `agent.policy_gradient_update`
- [policy_gradient_update](https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/capture_chess/agent.py#L172)
    - `targets = np.zeros((n_steps, 64, 64))`
    - loss = [modified_crossentropy](https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/capture_chess/agent.py#L9)

# todo
- understand and fix reinforce.py
- do the actor critic example from RLC part 4
- maybe
    - https://tqdm.github.io/ progress bar to training and eval
    - perf: chess is slow. multiproc?
    - does sb3 do better?

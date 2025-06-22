# Chess

Inspired by https://www.kaggle.com/code/arjanso/reinforcement-learning-chess-5-tree-search
Code at https://github.com/arjangroen/RLC

# Quick start
```sh
uv sync                   # install deps
./precommit.sh            # run all linters + tests
uv run bot_showdown.py    # vs all bots against each other
```

# todo
- make & train small net
    - plan
        - DONE write implementation plan
        - DONE do human parts first
        - check plan with claude
        - execute plan
            - feed plan to claude in chunks
- maybe: add small reward for piece captures
- evaluate vs opponents
    - random
    - greedy
    - andoma
    - (maybe) mcts with random rollout
    - (maybe) alphazero
- (maybe) impl in sb3, use optuna
- make & train big net
- (maybe): mcts improvement: save previous game tree?


# My notes on 'RLC'
- V-learning: just learn state values
- V network architecture is "quite arbitrary and can probably be optimized"
- moves are planned with MCTS

## training
- https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/agent.py#L21
`opponent = agent.GreedyAgent()` Greedily chooses highest material value.

- [network](https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/agent.py#L43)
`player = agent.Agent(lr=0.0005,network='big')`

- [learn](https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/learn.py#L18)
- play game `iters` times
    - [play_game](https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/learn.py#L73)
        - max 80 halfmoves
        - move = white move:
            - mcts move after X games, else random legal move
            - tree = self.mcts
        - move = black move:
            - pick highest value next state (greedy)
    - `reward`: gives small reward for piece captures
        - [saves state, samples minibatch, trains agent](https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/learn.py#L152)
            - self.update_agent -> agent.TD_update
            - [agent.TD_update](https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/agent.py#L205)
                - straightforward batch state + next state + reward fit


## other implementation details
- instead of copying the board state, keeps track of moves & reverts board state
  with:
```py
self.env.board.pop()
self.env.init_layer_board()
```

- color: 1 = white, -1 = black: https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/tree.py#L61
- note that pychess `chess.Color` uses True = white, False = black

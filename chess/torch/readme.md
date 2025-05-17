# Chess

Inspired by https://www.kaggle.com/code/arjanso/reinforcement-learning-chess-5-tree-search
Code at https://github.com/arjangroen/RLC

# Quick start
```sh
uv sync
```

# todo
- fix test_mcts hangs
- make & train small net
- evaluate vs opponents
    - random
    - greedy
    - andoma
    - (maybe) mcts with random rollout
    - (maybe) alphazero
- make & train big net
- maybe: perf: don't copy board in mcts, use undo + move history
- (maybe): mcts improvement: save previous game tree?


# My notes on 'RLC'
- V-learning: just learn state values
- V network architecture is "quite arbitrary and can probably be optimized"
- moves are planned with MCTS

Code exploration:

- https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/agent.py#L21
`opponent = agent.GreedyAgent()` Greedily chooses highest material value.

- [network](https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/agent.py#L43)
`player = agent.Agent(lr=0.0005,network='big')`

- [learn](https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/learn.py#L18)
- play game `iters` times
    - [play_game](https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/learn.py#L73)
        - move = white move:
            - mcts move after X games, else random legal move
            - tree = self.mcts
        - move = black move:
            - pick highest value next state (greedy)
    - gives small reward for piece captures

- instead of copying the board state, keeps track of moves & reverts board state
  with:
```py
self.env.board.pop()
self.env.init_layer_board()
```

- color: 1 = white, -1 = black: https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/tree.py#L61
- note that pychess `chess.Color` uses True = white, False = black

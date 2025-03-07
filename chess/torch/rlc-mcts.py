"""
From https://www.kaggle.com/code/arjanso/reinforcement-learning-chess-5-tree-search

RLC code at https://github.com/arjangroen/RLC

- V-learning: just learn state values
- V network architecture is "quite arbitrary and can probably be optimized"
- moves are planned with MCTS

Explanation of:

# https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/agent.py#L21
opponent = agent.GreedyAgent()
    # .predict returns sum of material value

env = environment.Board(opponent, FEN=None)

# https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/agent.py#L43
player = agent.Agent(lr=0.0005,network='big')
    # big network:
    # optimizer = RMSprop
    # network: a bunch of conv2ds then dense layers with sigmoid activation.
    #          total params: 300k
    #          I'll get chatGPT to convert this to torch for me.

# https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/learn.py#L18
# "Chess algorithm that combines bootstrapped monte carlo tree search with Q Learning"
# is this similar to alphazero?
learner = learn.TD_search(env, player,gamma=0.9,search_time=0.9)

node = tree.Node(learner.env.board, gamma=learner.gamma)

# https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/learn.py#L49
learner.learn(iters=10000,timelimit_seconds=3600,c=10)  # c = fix model interval (update target network, double learning)
    # play game `iters` times
    # play_game: https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/learn.py#L73
    #   move = white move:
    #     mcts move after X games, else random legal move
    #        tree = self.mcts
    #   move = black move:
    #     pick highest value next state
    #   env step(move)
    #   update mcts tree to selected child action: tree = tree.children[max_move]
    #   save state, reward, td error etc. for batch learning
    #   self.update_agent() every 10 turns: update NN

# mcts = self.mcts: https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/learn.py#L215
    # add children with value prediction (from NN) to current node
    # select best child node:
    #   node.select:https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/tree.py#L50

# env step(move): https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/environment.py#L71
    # returns done, reward
    # reward = capture_factor*capture_value + (1 for win, -1 for loss else 0)
    #   capture_factor = 0.01
"""

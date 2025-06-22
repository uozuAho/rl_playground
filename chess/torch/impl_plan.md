- what I already have implemented
    - chess environment in lib/env.py
    - random agent in lib/agent.py
    - MCTS algorithm in lib/mcts.py with
        - client-supplied value function for estimating a chess state (or random rollout by default)
        - configurable number of simulations
    - (not in this project) an agent with hard-coded position evaluation
- implement a greedy agent:
    - use ChessBoard.state_np as input to the neural network
    - it has a small simple value neural network for fast training. The network
      estimates the value of the chess board state, which corresponds to the
      probability of winning from that state. The network does not output action
      values.
    - train against a supplied agent that implements ChessAgent in lib/agent.py
    - makes greedy moves based on the output of the network
    - updates the network using experience replay buffer
    - uses a target network with soft updates
    - uses gradient clipping
- add an agent evaluator that logs the following:
    - win rate vs various opponents
    - average game length
    - evaluation loss
- add the ability to plot output from the evaluator
- check the greedy agent using the evaluator
- add self-play capability to the agent
- add the ability to save/load the agent's model
- check the small neural network: can it approximate a known fixed value function?
- implement a mcts agent
    - copy paste the greedy agent implementation
    - change greedy moves for mcts moves
- check mcts agent: does it outperform the greedy agent?
- implement a larger neural network
    - use techniques appropriate to chess. For example, convolutional layers to
      capture the 2d state of the game.
    - check: does it approximate a fixed value function better than the smaller
      network?
- use the larger neural net with the mcts agent

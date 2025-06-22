- HUMAN: add layer_board to chess env: https://github.com/arjangroen/RLC/blob/master/RLC/real_chess/environment.py
- implement a greedy agent:
    - has a small neural network for fast training
    - makes greedy moves based on the output of the network
    - updates the network using random-sampled minibatches
    - uses double learning with soft updates
    - uses gradient clipping
- check the greedy agent: does it learn?
- check the small neural network: can it approximate a known fixed value function?
- implement a mcts agent
    - copy paste the greedy agent implementation
    - change greedy moves for mcts moves
- check mcts agent: does it outperform the greedy agent?
- implement a larger neural network
    - check: does it approximate a fixed value function better than the smaller
      network?
- use the larger neural net with the mcts agent


- maybe? use TTT as reference
    - **todo** copy these files to ./reference/
    - tab mcts for mcts usage
    - nn greedy for nn training (needs double learning, not implemented)

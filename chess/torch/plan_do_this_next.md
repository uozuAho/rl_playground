- implement a greedy chess agent with the following attributes:
    - uses ChessBoard.state_np as input to the neural network
    - has a small simple value neural network for fast training. The network
      estimates the value of the chess board state, which corresponds to the
      probability of winning from that state. The network does not output action
      values.
    - makes greedy moves based on the output of the network
    - updates the network using experience replay buffer
    - uses a target network with soft updates
    - uses gradient clipping
    - trains against a supplied agent that implements ChessAgent in lib/agent.py

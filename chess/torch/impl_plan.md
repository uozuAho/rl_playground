- implement a greedy agent:
    - it has a small neural network for fast training. The network estimates the
      value of the chess board state, which corresponds to the probability of
      winning from that state.
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

## Suggestions and Improvements:

### Technical Corrections:
- For the greedy agent, consider using experience replay buffer instead of just
  random minibatches for more stable learning
- Double learning typically refers to Double DQN - clarify if you mean target
  networks with soft updates
- Add evaluation metrics beyond just "does it learn" - track win rate, average
  game length, evaluation loss over time

### Implementation Details to Consider:
- Data representation: How will you encode the chess board state for the neural
  network?
- Action space: How will you handle the large action space in chess (~4000
  possible moves)?
- Training data: Will you use self-play, human games, or engine games for
  initial training?
- Network architecture: Consider starting with a simple fully connected network,
  then move to convolutional layers

### Additional Steps to Add:
- Implement baseline random agent for comparison
- Add proper logging and visualization of training progress
- Implement proper chess move validation and game termination
- Add checkpointing to save/load trained models
- Consider adding an intermediate step: greedy agent with basic position
  evaluation features before pure neural network

### MCTS Specific Considerations:
- UCB1 vs other selection policies
- Number of simulations per move
- How to integrate neural network evaluations with MCTS tree search
- Balancing exploration vs exploitation in the tree

### Evaluation Framework:
- Tournament-style evaluation between different agents
- Evaluation against standard chess engines at different skill levels
- Opening book usage considerations

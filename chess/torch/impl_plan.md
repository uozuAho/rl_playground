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
- add checkpointing to the agent
    - save models at regular intervals during training
    - make it easy to load a saved model
- check the small neural network: can it approximate a known fixed value function?
- (maybe) Add board symmetries (rotations, reflections) to increase training
  data diversity
- (maybe) Implement adaptive learning rates or warmup schedules for better
  convergence
- implement a mcts agent
    - copy paste the greedy agent implementation
    - change greedy moves for mcts moves
- check mcts agent: does it outperform the greedy agent?
- implement a larger neural network
    - use techniques appropriate to chess. For example, convolutional layers to
      capture the 2d state of the game.
    - check: does it approximate a fixed value function better than the smaller
      network?
    - (maybe) Regularization: Add dropout or weight decay to prevent
      overfitting, especially for the larger network
    - (maybe) Consider adding batch norm layers to stabilize training
- use the larger neural net with the mcts agent
- (maybe) ELO rating system: Track relative strength changes over time instead
  of just win rates

(maybe, if not learning well)
- Multi-step returns: Use n-step TD targets instead of just 1-step for better
  value estimation
- Curriculum learning: Start with simplified positions (fewer pieces) and
  gradually increase complexity
- Debugging and Analysis:
    - Value function visualization: Plot learned values across different game
      phases
    - Move probability heatmaps: Visualize what the network considers for each
      position
    - Training diagnostics: Monitor gradient norms, activation statistics, and
      loss components
- Hyperparameter tuning: Systematically search learning rates, network sizes,
  and MCTS parameters

(maybe? suggestion by claude, dunno what it means)
- Prioritized experience replay: Weight important experiences more heavily
  during training
- MCTS Enhancements:
    - UCB1 tuning: Experiment with different exploration constants
    - Progressive widening: Limit action space exploration early in search
    - Transposition tables: Cache evaluations of repeated positions

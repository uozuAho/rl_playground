# C# chess agents, experiments etc

Uses [TorchSharp](https://github.com/dotnet/TorchSharp) and
[Chess-Coding-Adventure](https://github.com/SebLague/Chess-Coding-Adventure)

The chess implementation is much faster than pychess (at least 100x).
TorchSharp seems decent, but requires manual memory management when
using the GPU :(

# quick start
- install dotnet 8, make
- `git clone --recurse-submodules`

```sh
make pc  # restoring packages on first run may take some time - torchsharp is big
cd cschess.experiments
dotnet run chess gpu
```

# Todo
- WIP implement alphazero single thread first
  - WIP do one/few training steps ASAP. does training work?
    - WIP learner
      - finish update_net, see if u can train it
      - later: merge state encoder/codec. they depend on each other
  - WIP self play games
    - test for state immutability, see todos
    - test heat, heat dict
    - test self play games - does returned trajectory make sense?
  - todo: unskip batch eval test. Still NaN?
- check training speed, maybe optimise, maybe make multithread version
- maybe make mcts agents respect timeout
- bot ranker
  - add andoma?
  - maybe: report avg time per move per agent
- maybe: handle all possible moves. See LegalMoves - mine filters out dupes
  that have same from-to squares
## old todos
- greedy nn bot
  - DONE test small run on cpu
  - DONE print stats every episode (currently pretty slow)
  - DONE train against coding adventure bot, eval against random
  - train/tweak greedy bot. is it learning/improving?
    - do long training run
      - plot/log stats to file for later plot
- maybe: optimise. reuse tensors? can't find any docs. give it a try. See
  torch_vs_torchsharp. Only do this if training slows down dramatically
  with more episodes.
  - pytorch caches, torchsharp doesn't? See https://docs.pytorch.org/docs/stable//notes/cuda.html#cuda-memory-management
- add save, load, checkpointing to greedy bot
- (automatically?) add saved bots to bot tournament
- (automatically?) log tournament results
    - pretty print in bot strength order
- add self play to greedy bot
- add planning to greedy bot
    - mcts or coding adventure search
- possible bot improvements
    - use a larger net
        - copilot suggestion: Add dropout or weight decay to prevent
          overfitting, especially for the larger network
        - plot training data. scottplot? csv + matplotlib?
    - check: does it approximate a fixed value function better than the smaller
      network?
    - add board symmetries (rotations, reflections) to training
    - implement adaptive learning rates or warmup schedules
- (maybe, if not learning well)
    - Multi-step returns: Use n-step TD targets instead of just 1-step for
      better value estimation
    - Curriculum learning: Start with simplified positions (fewer pieces) and
      gradually increase complexity
    - Debugging and Analysis:
        - Value function visualization: Plot learned values across different
        game phases
        - Move probability heatmaps: Visualize what the network considers for
        each position
        - Training diagnostics: Monitor gradient norms, activation statistics,
        and loss components
    - Hyperparameter tuning: Systematically search learning rates, network
      sizes, and MCTS parameters

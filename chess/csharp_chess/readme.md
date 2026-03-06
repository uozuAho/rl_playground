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

# opt log
```cs
const int numIterations = 2;
const int nParallelGames = 4;
const int nMctsSims = 60;
var evaluaotr = UnifBatchEval
```

# Todo
- az: train. does it improve?
    - WIP perf: still p slow. do low hanging fruit. not multithread yet
        - py maxed out about 30 steps/sec with nn 2 48, mcts 60, multiprocess
        - current C# nn 2 48, mcts 60, single thread: 30-50 steps/sec, 4-8 parallel games
        - profile notes:
            - most time in net.forward + cpu/gpu data transfer
        - ideas
            - game queue instead of batches of games - keep inference batch size the same
            - maybe multithread before/after eval
            - DONE: profile with noop evaluator. find non-gpu related issues
    - check pol val loss - does it improve?
        - if looks ok, eval vs random opponent
    - if no improvement, test basics, eg
        - heat, heat dict
        - self play final reward is correct for win, loss, draw scenarios
        - mask invalid actions
        - maybe: check for wins/losses during training. only seeing draws?
            - maybe:
                - add capture reward
                - simplify rules
                - train with existing replays
## maybe
- az: chess perf
    - az: creating/copying board during pmcts is still pretty heavy. preallocate? obj pool?
    - make mcts agents respect timeout
    - bot ranker
        - add andoma?
        - maybe: report avg time per move per agent
    - maybe: az: handle all possible moves. See LegalMoves - mine filters out dupes
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

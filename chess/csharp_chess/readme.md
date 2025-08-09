# C# chess agents, experiments etc

Uses [TorchSharp](https://github.com/dotnet/TorchSharp) and
[Chess-Coding-Adventure](https://github.com/SebLague/Chess-Coding-Adventure)

The chess implementation is much faster than pychess (at least 100x).
TorchSharp seems decent, but requires manual memory management when
using the GPU :(

# quick start
- install dotnet 8
- generate a file of FENs and score using ../torch/train_value_network.py
- put the file here, call it joe

```sh
dotnet test
dotnet run joe

dotnet csharpier format .   # format code
```

# Todo
- greedy nn bot
  - DONE test small run on cpu
  - DONE print stats every episode (currently pretty slow)
  - DONE train against coding adventure bot, eval against random
  - WIP optimise
    - log
      - cpu: 2.33 games/sec, 71.25 steps/sec
      - gpu: 2.64 games/sec, 87.70 steps/sec
      - remove individial tensor element get/set in board2tensor
        - cpu: 2.77 games/sec, 89.85 steps/sec
        - gpu: 3.21 games/sec, 107.11 steps/sec
      - sequential model
        - cpu: 2.52 games/sec, 87.03 steps/sec
        - gpu: 3.12 games/sec, 114.09 steps/sec
      - **todo**
        - convert to tensor as late as possible, use float arrays instead
        - wrap tensor calcs in `using (var d = torch.NewDisposeScope())`, see https://github.com/dotnet/TorchSharpExamples/blob/main/src/CSharp/CSharpExamples/MNIST.cs#L107
          - and https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/memory.md
        - maybe: reuse tensors? can't find any docs. give it a try. See
          torch_vs_torchsharp. Only do this if training slows down dramatically
          with more episodes.
  - train/tweak greedy bot. is it learning/improving?
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


# how does coding adventure bot work
- engineUCI
  - player = bot
  - OnMoveChosen: respond "bestmove <move>"
  - new game
    - bot.NotifyNewGame
      - searcher.ClearForNewPosition
  - ProcessPositionCommand msg
    - if msg contains "startpos"
      - bot.SetPosition(regular new game)
  - ProcessGoCommand msg
    - if "movetime" in msg
      - bot.ThinkTimed(time)
    - else
      - bot.ChooseThinkTime -> ThinkTimed
  - if msg = quit
    - bot.Quit

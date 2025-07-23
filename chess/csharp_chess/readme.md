# C# chess agents, experiments etc

Uses [TorchSharp](https://github.com/dotnet/TorchSharp) and [ChessLib](https://github.com/rudzen/ChessLib/)

# todo
- WIP ValueNetworkTrainer
  - compare to py. speed, accuracy
    - py: gpu, 7s
      - Mean Squared Error (MSE): 0.0017
        Mean Absolute Error (MAE): 0.0292
        Pearson Correlation: 0.9867
    - cs: cpu, 20s
      - mse: 0.0024842160542706522
        mae: 0.03746429003634781
        correlation: 0.990030127810511
    - cs: GPU: WIP
      - memory issues: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/memory.md
- if above wrapper good:
  - remove ms dep injection dependency from chess.game
  - remove chesslib game wrapper
  - remove chesslib submodule
  - update these docs
  - maybe: fix build warnings

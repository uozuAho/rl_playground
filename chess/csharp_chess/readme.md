# C# chess agents, experiments etc

Uses [TorchSharp](https://github.com/dotnet/TorchSharp) and [ChessLib](https://github.com/rudzen/ChessLib/)

# todo
- finish ValueNetworkTrainer: blocked needing fen->game->input tensor
  - compare to py. speed, accuracy
    - try gpu
- if above wrapper good:
  - remove ms dep injection dependency from chess.game
  - remove chesslib game wrapper
  - remove chesslib submodule
  - update these docs
  - maybe: fix build warnings

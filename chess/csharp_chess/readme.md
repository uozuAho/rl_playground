# C# chess agents, experiments etc

Uses [TorchSharp](https://github.com/dotnet/TorchSharp) and [ChessLib](https://github.com/rudzen/ChessLib/)

# todo
- WIP: wrap game, add test: play random game
  - use chess coding adventure impl
- finish ValueNetworkTrainer: blocked needing fen->game->input tensor
  - try cpu & gpu
- if above wrapper good:
  - remove ms dep injection dependency from chess.game
  - remove chesslib game wrapper
  - remove chesslib submodule
  - update these docs
  - maybe: fix build warnings

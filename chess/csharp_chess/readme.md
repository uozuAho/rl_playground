# C# chess agents, experiments etc

Uses [TorchSharp](https://github.com/dotnet/TorchSharp) and [Chess-Coding-Adventure](https://github.com/SebLague/Chess-Coding-Adventure)

The chess implementation is much faster than pychess (at least 100x).
TorchSharp seems decent, but requires manual memory management when
using the GPU :(

# quick start
- generate a file of FENs and score using ../torch/train_value_network.py
- put the file here, call it joe
- `dotnet run joe`

# todo
- maybe: fix build warnings

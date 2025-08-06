# TorchSharp vs pytorch

Testing ground to compare pytorch + TorchSharp

# Quick start
```sh
cd pytorch
uv sync
uv run simple.py
uv run chess_rl_sim.py
cd..
cd csharp
dotnet run simple
dotnet run chess
```

# todo
- try to make C# chess sim as fast as py
    - cpu is close enough
    - gpu is ~70% py speed

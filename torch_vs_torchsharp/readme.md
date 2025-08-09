# TorchSharp vs pytorch

Testing ground to compare pytorch + TorchSharp

# Quick start
```sh
cd pytorch
uv sync
uv run simple.py
uv run chess_rl_sim.py
uv run python -m cProfile chess_rl_sim.py > profile.txt
cd..
cd csharp
dotnet run simple
dotnet run chess
dotnet-trace collect -o chess_cpu.nettrace -- ./bin/Debug/net8.0/csharp chess cpu
dotnet trace report chess_cpu.nettrace topN -n 20 > profile_cpu.txt
```

# todo
- maybe: try reusing tensors. Ask ChatGPT how to copy `float[,,]` into an
  existing tensor. I haven't tried it but it looks possible. Lots of code
  though, and may run into more issues with unsafe memory usage and reusing
  tensors. Not worth it for a 20-30% speedup. I doubt it will end up a lot
  faster than python.

# notes on perf
- gpu originally ~80% py speed for chess
    - py Done training in 2.57s. 38.85 eps/sec, 1981.55 moves/sec
    - C# Done training in 3.14s. 31.86 eps/sec, 1624.79 moves/sec
    - use NewDisposeScope per episode speeds things up, but slows down with more
      episodes. Using a single NewDisposeScope for all training eventually runs
      out of GPU memory. Using NewDisposeScope every X episodes works, but slows
      down with more episodes. For some reason, disposing tensors gets slower
      the longer training runs.

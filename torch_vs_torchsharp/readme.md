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
- try to make C# chess sim as fast as py
    - cpu
        py Done training in 3.74s. 26.72 eps/sec, 1362.62 moves/sec
        C# Done training in 4.29s. 23.29 eps/sec, 1187.64 moves/sec
        use NewDisposeScope per episode:
        C# Done training in 3.69s. 27.11 eps/sec, 1382.78 moves/sec
    - gpu originally ~80% py speed
        py Done training in 2.57s. 38.85 eps/sec, 1981.55 moves/sec
        C# Done training in 3.14s. 31.86 eps/sec, 1624.79 moves/sec
        C# use NewDisposeScope per episode:
            Done training in 2.55s. 39.19 eps/sec, 1998.60 moves/sec
            This slows down with more episodes.
           use NewDisposeScope for all of training:
            Done training in 9.37s. 42.68 eps/sec, 2176.46 moves/sec
            This eventually runs out of memory on the GPU.
           using a hybrid approach: NewDisposeScope every X episodes
            faster, but still slows down with number of episodes. at 2000eps, dispose every 100:
            Done training in 62.24s. 32.13 eps/sec, 1638.76 moves/sec
                I noticed that disposing tensors takes longer over time. why?
        python maintains ~40eps/sec over 2000 eps:
            Done training in 48.15s. 41.54 eps/sec, 2118.52 moves/sec

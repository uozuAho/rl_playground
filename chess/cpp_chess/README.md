# C++ chess

Faster than python chess! Uses LeelaChessZero chess implementation.


# Quick start
Prerequisites:
- linux + `sudo apt-get install build-essential`. This should hopefully get you:
    - a C++20 capable compiler
    - make, cmake

Then:
```sh
./run.sh test       # run tests
./run.sh --release  # run whatever's in main.cpp (optimised build)
./run.sh            # debug run, probably very slow!
./run.sh --rebuild  # if something's going wrong during build, try a rebuild
```

# Debugging
- run ./run.sh test to build in debug mode (there's no build-only option at the moment)
- use vscode debugger with C++ extension

# to do
- copy greedy nn bot from python. goal: play games + train much faster
- make better bots? mcts etc.
    - use go/C#? I hate C++
- (maybe) try gpu for speedup. train_value_net is slower than python gpu
- maybe: perf: andoma: implement move ordering
- maybe: perf: general profile + optimise

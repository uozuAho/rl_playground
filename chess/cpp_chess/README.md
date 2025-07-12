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
- try training a torch model. Prompt:

Port the python code in train_value_network.py to C++. Add the C++ code to
EvalApproximator. The python code trains a neural network to approximate a chess
board evaluation function. It then evaluates how well the network approximates
the evaluation function. In this case, our evaluation function is evaluate_board
in eval.cpp.

- copy greedy nn bot from python
- maybe: perf: andoma: implement move ordering
- maybe: perf: general profile + optimise

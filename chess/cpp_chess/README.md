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
- WIP try training a torch model
    - DONE gen code
    - DONE try running it, fix compile/run
    - WIP: why worse than python version?
        - WIP review nets, training, eval: doing same thing?
            - DONE nets are same
            - DONE gendata: same
            - DONE: train: cpp doesn't shuffle data. Doesn't explain bad stats.
        - DONE: generate sample data in file to use in both
        - cpp: read datafile, confirm cpp mich eval scores are same
        - cpp: train + test from datafile
- copy greedy nn bot from python
- maybe: perf: andoma: implement move ordering
- maybe: perf: general profile + optimise

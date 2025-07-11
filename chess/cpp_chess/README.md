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
./run.sh            # run whatever's in main.cpp
./run.sh --release  # run it fast (optimised build)
./run.sh --rebuild  # if something's going wrong during build, try a rebuild
```

# Debugging
- in CMakelists.txt, comment out `set(CMAKE_BUILD_TYPE Release)` and uncomment
  the next 3 lines
- run ./run.sh to build
- use vscode debugger. You can debug main or the tests.

# to do
- try training a torch model
- maybe: perf: andoma: implement move ordering

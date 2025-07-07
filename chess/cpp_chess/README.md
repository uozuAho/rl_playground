# C++ chess

Faster than python chess! Uses LeelaChessZero chess implementation.


# Quick start
Prerequisites:
- linux + `sudo apt-get install build-essential`. This should hopefully get you:
    - a C++20 capable compiler
    - make, cmake

Then:
```sh
./run.sh test  # run tests
./run.sh       # run whatever's in main.cpp
./run.sh --rebuild  # if something's going wrong during build, do a rebuild
```

# to do
- WIP: impl miech board eval
    - DONE get test running
    - compare tests with py tests
    - get all tests passing
    - review eval code, do easy optimisations
- WIP: implement andoma
    - WIP add miech evaluation
    - implement move ordering
    - write bot fight to check win rates

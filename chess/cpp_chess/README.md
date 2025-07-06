# cpp_chess_leela_random_agents

This project runs two random chess agents against each other using
LeelaChessZero's chess impl.

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
- fix run script when no args passed
- WIP: impl miech board eval
- WIP: implement andoma
    - WIP add miech evaluation
    - implement move ordering
    - write bot fight to check win rates

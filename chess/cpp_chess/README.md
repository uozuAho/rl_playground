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
- impl miech board eval
    - add required interfaces to board wrapper
    - need piece_color(square)
- WIP: implement andoma
    - implement stubs in chess wrapper
    - add miech evaluation
    - implement move ordering

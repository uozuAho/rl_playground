# cpp_chess_leela_random_agents

This project runs two random chess agents against each other using LeelaChessZero's C++ chess interface (not UCI).

## Project Structure
- `src/` - Source files
- `include/` - Header files
- `CMakeLists.txt` - Build configuration

## Getting Started

### 1. Clone LeelaChessZero as a submodule
```
git submodule add https://github.com/LeelaChessZero/lc0.git external/lc0
```

### 2. Build LeelaChessZero (chess core only)
Follow instructions in `external/lc0/README.md` to build the chess core, or link to its source in your CMake project.

### 3. Build this project
```
mkdir build
cd build
cmake ..
make
```

### 4. Run
```
./cpp_chess
```

## Extending Agents
- Implement new agents in `src/` and inherit from the `Agent` interface.
- Register new agents in `main.cpp`.

## Notes
- Uses LeelaChessZero's `board.h` for board representation and move generation.
- Logging is minimal for speed; enable as needed in code.

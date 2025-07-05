# cpp_chess_leela_random_agents

This project runs two random chess agents against each other using
LeelaChessZero's chess impl.


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

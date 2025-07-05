cmake_minimum_required(VERSION 3.10)
project(cpp_chess)

set(CMAKE_CXX_STANDARD 17)

# Add LeelaChessZero as a submodule or external project
# (Instructions in README.md)

include_directories(src/mystuff/include)
include_directories(lc0chess/chess)

add_executable(cpp_chess
    src/mystuff/cpp/main.cpp
    src/mystuff/cpp/game.cpp
    src/mystuff/cpp/agent_random.cpp
    src/mystuff/cpp/leela_board_wrapper.cpp

    src/lc0chess/chess/board.cc
    src/lc0chess/chess/board_test.cc
    src/lc0chess/chess/gamestate.cc
    src/lc0chess/chess/position.cc
    src/lc0chess/chess/position_test.cc
    src/lc0chess/chess/uciloop.cc
)

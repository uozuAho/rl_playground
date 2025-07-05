cmake_minimum_required(VERSION 3.10)
project(cpp_chess)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DNO_PEXT)

include_directories(src/lc0chess)
include_directories(src/mystuff/include)

add_executable(cpp_chess
    src/lc0chess/chess/board.cc
    src/lc0chess/chess/gamestate.cc
    src/lc0chess/chess/position.cc

    src/mystuff/cpp/main.cpp
    src/mystuff/cpp/game.cpp
    src/mystuff/cpp/agent_random.cpp
    src/mystuff/cpp/leela_board_wrapper.cpp
)

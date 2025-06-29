cmake_minimum_required(VERSION 3.10)
project(cpp_chess_leela_random_agents)

set(CMAKE_CXX_STANDARD 17)

# Add LeelaChessZero as a submodule or external project
# (Instructions in README.md)

include_directories(include)

add_executable(cpp_chess
    src/main.cpp
    src/game.cpp
    src/agent_random.cpp
    src/leela_board_wrapper.cpp
)

# Link LeelaChessZero libraries here when integrated
# target_link_libraries(cpp_chess lc0_chess)

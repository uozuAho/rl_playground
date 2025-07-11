cmake_minimum_required(VERSION 3.10)
project(cpp_chess)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimised build, can't debug
# set(CMAKE_BUILD_TYPE Release)

# Debug build, can debug! Uncomment the following 3 lines
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")

add_definitions(-DNO_PEXT)

include_directories(src/lc0chess)
include_directories(src/mystuff/include)

add_executable(cpp_chess
    src/lc0chess/chess/board.cc
    src/lc0chess/chess/gamestate.cc
    src/lc0chess/chess/position.cc
    src/lc0chess/utils/logging.cc

    src/mystuff/cpp/main.cpp
    src/mystuff/cpp/eval.cpp
    src/mystuff/cpp/agent_random.cpp
    src/mystuff/cpp/andoma_agent.cpp
    src/mystuff/cpp/leela_board_wrapper.cpp
)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(allTests
    test/test_andoma.cpp
    test/test_leela_board_wrapper.cpp
    test/test_eval.cpp
    src/mystuff/cpp/leela_board_wrapper.cpp
    src/mystuff/cpp/agent_random.cpp
    src/mystuff/cpp/andoma_agent.cpp
    src/mystuff/cpp/eval.cpp
    src/lc0chess/chess/board.cc
    src/lc0chess/chess/gamestate.cc
    src/lc0chess/chess/position.cc
    src/lc0chess/utils/logging.cc
)

target_include_directories(allTests PRIVATE src/mystuff/include src/lc0chess)
target_link_libraries(allTests gtest_main)

add_test(NAME allTests COMMAND allTests)

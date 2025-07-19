cmake_minimum_required(VERSION 3.10)
project(cpp_chess)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/libtorch/share/cmake/Torch" ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

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
    src/mystuff/cpp/bot_fight.cpp
    src/mystuff/cpp/agent_random.cpp
    src/mystuff/cpp/andoma_agent.cpp
    src/mystuff/cpp/leela_board_wrapper.cpp
    src/mystuff/cpp/torch_eval_approximator.cpp
    src/mystuff/cpp/greedy_chess_agent.cpp
)

target_link_libraries(cpp_chess "${TORCH_LIBRARIES}")

enable_testing()

# Only build allTests if TESTS_ENABLED is ON
option(TESTS_ENABLED "Build tests" OFF)

if(TESTS_ENABLED)
    include(FetchContent)
    FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

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
endif()

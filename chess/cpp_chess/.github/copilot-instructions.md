This project contains a C++ implementation of chess, and some chess playing
agents. It uses cmake to build, with cmake configuration in CMakeLists.txt.
There is a user-friendly run.sh script that hides the complexities of cmake.

All chess game implementation code is in src/lc0chess. Do not modify this unless
instructed to.

My code is under src/mystuff. This is where you should add code.
src/mystuff/include/leela_board_wrapper.h is a user-friendly wrapper around the
chess implementation in src/lc0chess.

Tests are under test/ and use google test.

#include <gtest/gtest.h>
#include "leela_board_wrapper.h"

// Sample test fixture for LeelaBoardWrapper
test(LeelaBoardWrapperTest, InitialState) {
    LeelaBoardWrapper board;
    EXPECT_FALSE(board.is_game_over());
    EXPECT_EQ(board.turn(), 0); // White to move
    EXPECT_EQ(board.color(), "white");
    auto moves = board.legal_moves();
    EXPECT_FALSE(moves.empty());
}

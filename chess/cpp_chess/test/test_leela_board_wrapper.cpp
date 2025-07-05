#include <gtest/gtest.h>
#include "leela_board_wrapper.h"
#include "chess/board.h" // For lczero::InitializeMagicBitboards

class LeelaBoardWrapperTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        lczero::InitializeMagicBitboards();
    }
};

TEST_F(LeelaBoardWrapperTest, InitialState) {
    LeelaBoardWrapper board;
    EXPECT_FALSE(board.is_game_over());
    EXPECT_EQ(board.turn(), 0); // White to move
    EXPECT_EQ(board.color(), "white");
    auto moves = board.legal_moves();
    EXPECT_FALSE(moves.empty());
}

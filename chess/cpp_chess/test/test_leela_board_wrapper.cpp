#include <gtest/gtest.h>
#include "leela_board_wrapper.h"
#include "chess/board.h"
#include "agent_random.h"

class LeelaBoardWrapperTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        lczero::InitializeMagicBitboards();
    }
};

TEST_F(LeelaBoardWrapperTest, InitialState) {
    LeelaBoardWrapper board;
    EXPECT_FALSE(board.is_game_over());
    EXPECT_EQ(board.turn(), 0);
    EXPECT_EQ(board.color(), "white");
    auto moves = board.legal_moves();
    EXPECT_FALSE(moves.empty());
}

TEST_F(LeelaBoardWrapperTest, PlayFullGame) {
    LeelaBoardWrapper board;
    RandomAgent white;
    RandomAgent black;
    Agent* agents[2] = {&white, &black};

    int turn = 0;
    int numHalfmoves = 0;
    while (!board.is_game_over()) {
        auto move = agents[turn]->select_move(board);
        board.make_move(move);
        turn = 1 - turn;
        numHalfmoves++;
    }

    EXPECT_TRUE(board.is_game_over());
}

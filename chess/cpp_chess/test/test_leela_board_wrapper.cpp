#include <gtest/gtest.h>
#include "leela_board_wrapper.h"
#include "chess/board.h"
#include "agent_random.h"

namespace mystuff {

class LeelaBoardWrapperTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        lczero::InitializeMagicBitboards();
    }
};

TEST_F(LeelaBoardWrapperTest, InitialState) {
    LeelaBoardWrapper board;
    EXPECT_FALSE(board.is_game_over());
    EXPECT_EQ(board.turn(), LeelaBoardWrapper::WHITE);

    auto moves = board.legal_moves();
    EXPECT_FALSE(moves.empty());

    EXPECT_TRUE(board.piece_at("a1").has_value());
    EXPECT_EQ(board.piece_at("a1").value(), lczero::kRook);
    EXPECT_EQ(board.color_at("a1"), LeelaBoardWrapper::WHITE);

    EXPECT_TRUE(board.piece_at("a7").has_value());
    EXPECT_EQ(board.piece_at("a7").value(), lczero::kPawn);
    EXPECT_EQ(board.color_at("a7"), LeelaBoardWrapper::BLACK);

    EXPECT_FALSE(board.piece_at("a6").has_value());
    EXPECT_EQ(board.color_at("a6"), 0);
}

TEST_F(LeelaBoardWrapperTest, HowDoSquaresWork) {
    LeelaBoardWrapper board;
    const auto square_0 = lczero::Square::FromIdx(0);
    const auto square_a1 = lczero::Square::Parse("a1");
    EXPECT_EQ(square_0, square_a1);

    EXPECT_EQ(board.piece_at(square_a1), lczero::kRook);
    EXPECT_EQ(board.piece_at(lczero::Square::Parse("a2")), lczero::kPawn);
    EXPECT_EQ(board.color_at(square_a1), LeelaBoardWrapper::WHITE);
    EXPECT_EQ(board.color_at(lczero::Square::Parse("a8")), LeelaBoardWrapper::BLACK);
}

TEST_F(LeelaBoardWrapperTest, HowDoMovesAndPositionsWork) {
    // as expected, but the implementation is tricky. You don't need to worry
    // about that!
    LeelaBoardWrapper board;
    EXPECT_EQ(board.turn(), LeelaBoardWrapper::WHITE);
    board.make_move("a2", "a3");
    EXPECT_TRUE(board.piece_at("a3").has_value());
    EXPECT_EQ(board.piece_at("a3").value(), lczero::kPawn);
    EXPECT_EQ(board.color_at("a3"), LeelaBoardWrapper::WHITE);

    EXPECT_EQ(board.turn(), LeelaBoardWrapper::BLACK);
    EXPECT_EQ(board.color_at("a7"), LeelaBoardWrapper::BLACK);
    board.make_move("a7", "a6");
    EXPECT_TRUE(board.piece_at("a6").has_value());
    EXPECT_EQ(board.piece_at("a6").value(), lczero::kPawn);
    EXPECT_EQ(board.color_at("a6"), LeelaBoardWrapper::BLACK);
    EXPECT_EQ(board.turn(), LeelaBoardWrapper::WHITE);
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

TEST_F(LeelaBoardWrapperTest, Copy) {
    LeelaBoardWrapper board1;
    board1.make_move("a2", "a3");

    // copy
    auto board2 = board1.copy();

    // copy == original
    EXPECT_EQ(board2.piece_at("a3").value(), lczero::kPawn);
    EXPECT_EQ(board1.fen(), board2.fen());

    // modify copy
    board2.make_move("a7", "a6");

    // copy != original
    EXPECT_EQ(board2.piece_at("a6").value(), lczero::kPawn);
    EXPECT_FALSE(board1.piece_at("a6").has_value());
    EXPECT_NE(board1.fen(), board2.fen());
}

} // namespace mystuff

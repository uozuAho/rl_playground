#include "gtest/gtest.h"
#include "eval.h"
#include "leela_board_wrapper.h"

using namespace mystuff;

static LeelaBoardWrapper board_from_fen(const std::string& fen) {
    return LeelaBoardWrapper::from_fen(fen);
}

TEST(EvalTest, StartingPositionIsEqual) {
    auto board = board_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    float eval = evaluate_board(board);
    EXPECT_NEAR(eval, 0.0, 0.1);
}

TEST(EvalTest, WhiteUpQueenIsHigh) {
    auto board = board_from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    float eval = evaluate_board(board);
    EXPECT_GT(eval, 7.0);
}

TEST(EvalTest, BlackUpQueenIsLow) {
    auto board = board_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1");
    float eval = evaluate_board(board);
    EXPECT_LT(eval, -7.0);
}

TEST(EvalTest, WhiteUpPawnIsSmall) {
    auto board = board_from_fen("rnbqkbnr/ppppppp1/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    float eval = evaluate_board(board);
    EXPECT_GT(eval, 0.7);
    EXPECT_LT(eval, 2.0);
}

TEST(EvalTest, BlackUpPawnIsSmall) {
    auto board = board_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP1/RNBQKBNR w KQkq - 0 1");
    float eval = evaluate_board(board);
    EXPECT_LT(eval, -0.7);
    EXPECT_GT(eval, -2.0);
}

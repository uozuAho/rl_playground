#include <tuple>
#include "gtest/gtest.h"
#include "eval.h"
#include "leela_board_wrapper.h"

using namespace mystuff;

static LeelaBoardWrapper board_from_fen(const std::string& fen) {
    return LeelaBoardWrapper::from_fen(fen);
}

struct EvalTestParam {
    std::string fen;
    float expected;
};

class EvalParameterizedTest : public ::testing::TestWithParam<EvalTestParam> {};

INSTANTIATE_TEST_SUITE_P(
    EvalCases,
    EvalParameterizedTest,
    ::testing::Values(
        EvalTestParam{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0},
        EvalTestParam{"8/8/8/8/8/8/8/8 w - - 0 1", 0},
        EvalTestParam{"8/8/8/8/8/8/8/K7 w - - 0 1", 20050},
        EvalTestParam{"8/8/8/8/8/8/8/k7 w - - 0 1", -19950},
        EvalTestParam{"8/8/8/8/8/8/8/4Q3 w - - 0 1", 895},
        EvalTestParam{"8/8/8/8/8/8/8/4q3 w - - 0 1", -895},
        EvalTestParam{"8/8/8/8/8/8/8/4P3 w - - 0 1", 100},
        EvalTestParam{"8/8/8/8/8/8/8/4p3 w - - 0 1", -100},
        EvalTestParam{"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", 0},
        EvalTestParam{"rnbq1bnr/ppppkppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQ - 3 4", 50}
    ),
    [](const ::testing::TestParamInfo<EvalTestParam>& info) {
        // Sanitize FEN for test name
        std::string name = info.param.fen;
        for (char& c : name) {
            if (!isalnum(c)) c = '_';
        }
        return name;
    }
);

TEST_P(EvalParameterizedTest, BoardEvalMatchesExpected) {
    const auto& param = GetParam();
    auto board = board_from_fen(param.fen);
    int eval = evaluate_board(board);
    EXPECT_EQ(eval, param.expected);
}

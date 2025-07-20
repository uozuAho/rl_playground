#include <gtest/gtest.h>
#include "leela_board_wrapper.h"
#include "chess/board.h"
#include "agent_random.h"
#include "andoma_agent.h"

namespace mystuff {

class AndomaTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        lczero::InitializeMagicBitboards();
    }
};

TEST_F(AndomaTest, PlayVsRandom) {
    LeelaBoardWrapper board;
    AndomaAgent andoma;
    RandomAgent rando;
    Agent* agents[2] = {&andoma, &rando};

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

} // namespace mystuff

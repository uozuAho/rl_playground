#include <iostream>
#include "agent_random.h"
#include "chess/board.h"

int main() {
    lczero::InitializeMagicBitboards();

    RandomAgent white = RandomAgent();
    RandomAgent black = RandomAgent();
    Agent* agents[2] = {&white, &black};
    LeelaBoardWrapper board = LeelaBoardWrapper();
    int turn = 0;
    while (!board.is_game_over()) {
        auto move = agents[turn]->select_move(board);
        board.make_move(move);
        turn = 1 - turn;
    }
    std::cout << "Result: " << board.result() << std::endl;
    return 0;
}

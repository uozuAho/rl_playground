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
    int numHalfmoves = 0;
    while (!board.is_game_over()) {
        auto move = agents[turn]->select_move(board);
        board.make_move(move);
        turn = 1 - turn;
        numHalfmoves++;
    }
    std::cout << "Result: " << board.result() << std::endl;
    std::cout << "Num moves: " << numHalfmoves << std::endl;
    return 0;
}

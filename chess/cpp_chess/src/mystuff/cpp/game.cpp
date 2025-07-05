#include "game.h"
#include <iostream>

Game::Game(Agent* white, Agent* black)
    : white_(white), black_(black) {
    board_.reset();
}

void Game::play() {
    Agent* agents[2] = {white_, black_};
    int turn = 0;
    while (!board_.is_game_over()) {
        auto move = agents[turn]->select_move(board_.board());
        board_.make_move(move);
        // move_history_.push_back(move.to_string()); // Implement as needed
        turn = 1 - turn;
    }
}

std::string Game::result() const {
    return board_.result();
}

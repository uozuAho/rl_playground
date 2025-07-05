#include "game.h"
#include <iostream>
#include "leela_board_wrapper.h"

Game::Game(Agent* white, Agent* black)
    : white_(white), black_(black), board_() {
}

void Game::play() {
    Agent* agents[2] = {white_, black_};
    int turn = 0;
    while (!board_.is_game_over()) {
        auto move = agents[turn]->select_move(board_);
        board_.make_move(move);
        turn = 1 - turn;
    }
}

std::string Game::result() const {
    return board_.result();
}

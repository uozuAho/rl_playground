#include "agent_random.h"
#include <random>
#include "leela_board_wrapper.h"

RandomAgent::RandomAgent() : rng(std::random_device{}()) {}

lcz::Move RandomAgent::select_move(const lcz::Board& board) {
    auto moves = lcz::legal_moves(board);
    if (moves.empty()) throw std::runtime_error("No legal moves");
    std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
    return moves[dist(rng)];
}

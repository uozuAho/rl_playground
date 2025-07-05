#include "agent_random.h"
#include <random>
#include "leela_board_wrapper.h"

RandomAgent::RandomAgent() : rng(std::random_device{}()) {}

lczero::Move RandomAgent::select_move(const lczero::ChessBoard& board) {
    auto moves = board.GenerateLegalMoves();
    if (moves.empty()) throw std::runtime_error("No legal moves");
    std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
    return moves[dist(rng)];
}

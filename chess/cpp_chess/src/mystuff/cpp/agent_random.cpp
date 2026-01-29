#include "agent_random.h"
#include <random>
#include "leela_board_wrapper.h"

namespace mystuff {

RandomAgent::RandomAgent() : rng(std::random_device{}()) {}

lczero::Move RandomAgent::select_move(const LeelaBoardWrapper& board) {
    auto moves = board.legal_moves();
    if (moves.empty()) throw std::runtime_error("No legal moves");
    std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
    return moves[dist(rng)];
}

} // namespace mystuff
